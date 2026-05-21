from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

SCHEMA = "azlite_capture_002_nonseparable_decomposition_v1"
SOURCE_SELECTION_SCORE_SCHEMA = "azlite_capture_002_selection_score_trace_v1"
SOURCE_TRACE_CADENCE_REVIEW_SCHEMA = "azlite_capture_002_trace_cadence_review_v1"
SOURCE_NONSEPARABLE_REVIEW_SCHEMA = "azlite_capture_002_nonseparable_review_v1"
ROW_ID = "capture_available-002"
CLASSIFICATION_DECISIONS = {
    "metric_co_movement": "stop_002_mechanism_not_isolated",
    "threshold_boundary_ambiguity": "write_002_confidence_band_spec",
    "signal_absent": "write_002_child_value_source_audit_spec",
    "decomposition_inconclusive": "stop_002_decomposition_inconclusive",
}
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
UNRESOLVED_SELECTION_DECISION = "write_002_unresolved_trace_review_spec"
ADEQUATE_CADENCE_DECISION = "continue_002_threshold_too_strict_check"
NONSEPARABLE_DECISION = "stop_002_unresolved"
BOUNDARY_BAND = 0.01


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose why capture 002 remained genuinely not separable"
    )
    parser.add_argument("--source-selection-score-artifact", type=Path, required=True)
    parser.add_argument("--source-threshold-review-artifact", type=Path, required=True)
    parser.add_argument(
        "--source-trace-cadence-review-artifact", type=Path, required=True
    )
    parser.add_argument(
        "--source-nonseparable-review-artifact", type=Path, required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _finite_number(value, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{context} must be finite numeric")
    return float(value)


def _integer(value, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def _classification_name(artifact: dict, *, context: str) -> str:
    classification = artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError(f"{context} classification must be an object")
    name = classification.get("classification")
    if not isinstance(name, str):
        raise ValueError(f"{context} classification.classification must be a string")
    return name


def _selection_source_identity(artifact: dict, *, context: str) -> dict:
    source_artifact = artifact.get("source_artifact")
    if not isinstance(source_artifact, dict):
        raise ValueError(f"{context} source_artifact must be an object")
    if source_artifact.get("row_id") != ROW_ID:
        raise ValueError(f"{context} source_artifact.row_id must be {ROW_ID}")
    selected_artifact = source_artifact.get("selected_artifact")
    if not isinstance(selected_artifact, dict):
        raise ValueError(
            f"{context} source_artifact.selected_artifact must be an object"
        )
    return {
        "row_id": source_artifact.get("row_id"),
        "reference_move": _integer(
            source_artifact.get("reference_move"),
            context=f"{context} source_artifact.reference_move",
        ),
        "full_search_selected_move": _integer(
            source_artifact.get("full_search_selected_move"),
            context=f"{context} source_artifact.full_search_selected_move",
        ),
        "selected_artifact": copy.deepcopy(selected_artifact),
    }


def _validated_selection_source_artifact(artifact: dict, *, context: str) -> dict:
    _selection_source_identity(artifact, context=context)
    return copy.deepcopy(artifact["source_artifact"])


def validate_selection_score_artifact_contract(
    artifact: dict,
    *,
    context: str,
    thresholds: dict,
) -> None:
    if artifact.get("schema") != SOURCE_SELECTION_SCORE_SCHEMA:
        raise ValueError(
            f"selection score artifact has wrong schema: expected {SOURCE_SELECTION_SCORE_SCHEMA}"
        )
    if artifact.get("insufficiency_reasons") != []:
        raise ValueError(f"{context} must have no insufficiency_reasons")
    if artifact.get("thresholds") != thresholds:
        raise ValueError(f"{context} has wrong threshold configuration")
    if _classification_name(artifact, context=context) != "unresolved":
        raise ValueError(f"{context} must remain unresolved")
    if artifact.get("decision") != UNRESOLVED_SELECTION_DECISION:
        raise ValueError(
            f"{context} unresolved decision must be {UNRESOLVED_SELECTION_DECISION}"
        )
    _selection_source_identity(artifact, context=context)
    for field in (
        "final_selected_minus_reference_q",
        "final_selected_minus_reference_selection_score",
        "final_selected_minus_reference_visit_share",
    ):
        _finite_number(artifact.get(field), context=f"{context} {field}")


def validate_trace_cadence_review_artifact_contract(artifact: dict) -> None:
    if artifact.get("schema") != SOURCE_TRACE_CADENCE_REVIEW_SCHEMA:
        raise ValueError(
            f"trace cadence review artifact has wrong schema: expected {SOURCE_TRACE_CADENCE_REVIEW_SCHEMA}"
        )
    if (
        _classification_name(artifact, context="trace cadence review artifact")
        != "cadence_adequate"
    ):
        raise ValueError(
            "trace cadence review artifact must represent adequate cadence"
        )
    if artifact.get("decision") != ADEQUATE_CADENCE_DECISION:
        raise ValueError(
            "trace cadence review artifact must represent adequate cadence"
        )
    count = artifact.get("unique_simulation_checkpoint_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(
            "trace cadence review artifact unique_simulation_checkpoint_count must be positive int"
        )
    checkpoints = artifact.get("unique_simulation_checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(
            "trace cadence review artifact unique_simulation_checkpoints must be a non-empty list"
        )
    for index, checkpoint in enumerate(checkpoints):
        _finite_number(
            checkpoint,
            context=f"trace cadence review artifact unique_simulation_checkpoints[{index}]",
        )


def validate_nonseparable_review_artifact_contract(artifact: dict) -> None:
    if artifact.get("schema") != SOURCE_NONSEPARABLE_REVIEW_SCHEMA:
        raise ValueError(
            f"nonseparable review artifact has wrong schema: expected {SOURCE_NONSEPARABLE_REVIEW_SCHEMA}"
        )
    if (
        _classification_name(artifact, context="nonseparable review artifact")
        != "genuinely_not_separable"
    ):
        raise ValueError("nonseparable review artifact must stop unresolved")
    if artifact.get("decision") != NONSEPARABLE_DECISION:
        raise ValueError("nonseparable review artifact must stop unresolved")
    if artifact.get("hypothesis") != "genuinely_not_separable":
        raise ValueError(
            "nonseparable review artifact hypothesis must be genuinely_not_separable"
        )


def _validate_trace_excerpt(
    trace_cadence_review_artifact: dict, source_identity: dict
) -> None:
    trace_capture_excerpt = trace_cadence_review_artifact.get("trace_capture_excerpt")
    if not isinstance(trace_capture_excerpt, dict):
        raise ValueError(
            "trace cadence review artifact trace_capture_excerpt must be an object"
        )
    if trace_capture_excerpt.get("row_id") != source_identity["row_id"]:
        raise ValueError(
            "trace cadence review artifact trace_capture_excerpt row_id must match selection source"
        )
    for field in ("reference_move", "full_search_selected_move"):
        if (
            _integer(
                trace_capture_excerpt.get(field),
                context=f"trace cadence review artifact trace_capture_excerpt.{field}",
            )
            != source_identity[field]
        ):
            raise ValueError(
                f"trace cadence review artifact trace_capture_excerpt {field} must match selection source"
            )


def _validate_selection_score_excerpt(
    trace_cadence_review_artifact: dict,
    default_selection_score_artifact: dict,
) -> None:
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


def _validate_cadence_input_path(
    trace_cadence_review_artifact: dict,
    *,
    source_selection_score_artifact_path: str,
) -> None:
    input_artifacts = trace_cadence_review_artifact.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise ValueError(
            "trace cadence review artifact input_artifacts must be an object"
        )
    if (
        input_artifacts.get("selection_score_artifact_path")
        != source_selection_score_artifact_path
    ):
        raise ValueError(
            "trace cadence review artifact input_artifacts selection_score_artifact_path must match source path"
        )


def _validate_nonseparable_paths(
    nonseparable_review_artifact: dict,
    *,
    source_selection_score_artifact_path: str,
    source_trace_cadence_review_artifact_path: str,
    source_threshold_review_artifact_path: str,
) -> None:
    input_artifacts = nonseparable_review_artifact.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise ValueError(
            "nonseparable review artifact input_artifacts must be an object"
        )
    expected_paths = {
        "source_selection_score_artifact_path": source_selection_score_artifact_path,
        "source_trace_cadence_review_artifact_path": source_trace_cadence_review_artifact_path,
        "source_threshold_review_artifact_path": source_threshold_review_artifact_path,
    }
    for key, expected_path in expected_paths.items():
        if input_artifacts.get(key) != expected_path:
            raise ValueError(
                f"nonseparable review artifact input_artifacts {key} must match source path"
            )


def _validate_nonseparable_summaries(
    default_selection_score_artifact: dict,
    trace_cadence_review_artifact: dict,
    threshold_review_artifact: dict,
    nonseparable_review_artifact: dict,
) -> None:
    thresholds_evaluated = nonseparable_review_artifact.get("thresholds_evaluated")
    if thresholds_evaluated != {
        "default_material_visit_share_margin": default_selection_score_artifact[
            "thresholds"
        ]["material_visit_share_margin"],
        "relaxed_material_visit_share_margin": threshold_review_artifact["thresholds"][
            "material_visit_share_margin"
        ],
    }:
        raise ValueError(
            "nonseparable review artifact thresholds_evaluated must match input thresholds"
        )

    expected_final_margin_summary = {
        "default_q_margin": default_selection_score_artifact[
            "final_selected_minus_reference_q"
        ],
        "default_selection_score_margin": default_selection_score_artifact[
            "final_selected_minus_reference_selection_score"
        ],
        "default_visit_share_margin": default_selection_score_artifact[
            "final_selected_minus_reference_visit_share"
        ],
        "relaxed_q_margin": threshold_review_artifact[
            "final_selected_minus_reference_q"
        ],
        "relaxed_selection_score_margin": threshold_review_artifact[
            "final_selected_minus_reference_selection_score"
        ],
        "relaxed_visit_share_margin": threshold_review_artifact[
            "final_selected_minus_reference_visit_share"
        ],
    }
    if (
        nonseparable_review_artifact.get("final_margin_summary")
        != expected_final_margin_summary
    ):
        raise ValueError(
            "nonseparable review artifact final_margin_summary must match input margins"
        )

    expected_source_snapshots = {
        "default_classification": default_selection_score_artifact["classification"],
        "cadence_classification": trace_cadence_review_artifact["classification"],
        "threshold_classification": threshold_review_artifact["classification"],
    }
    if (
        nonseparable_review_artifact.get("source_snapshots")
        != expected_source_snapshots
    ):
        raise ValueError(
            "nonseparable review artifact source_snapshots must match input classifications"
        )


def validate_input_chain_identity(
    default_selection_score_artifact: dict,
    trace_cadence_review_artifact: dict,
    threshold_review_artifact: dict,
    nonseparable_review_artifact: dict,
    *,
    source_selection_score_artifact_path: str,
    source_trace_cadence_review_artifact_path: str,
    source_threshold_review_artifact_path: str,
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

    _validate_trace_excerpt(trace_cadence_review_artifact, default_identity)
    _validate_selection_score_excerpt(
        trace_cadence_review_artifact, default_selection_score_artifact
    )
    _validate_cadence_input_path(
        trace_cadence_review_artifact,
        source_selection_score_artifact_path=source_selection_score_artifact_path,
    )
    _validate_nonseparable_paths(
        nonseparable_review_artifact,
        source_selection_score_artifact_path=source_selection_score_artifact_path,
        source_trace_cadence_review_artifact_path=source_trace_cadence_review_artifact_path,
        source_threshold_review_artifact_path=source_threshold_review_artifact_path,
    )
    _validate_nonseparable_summaries(
        default_selection_score_artifact,
        trace_cadence_review_artifact,
        threshold_review_artifact,
        nonseparable_review_artifact,
    )


def _first_support_summary(default_selection_score_artifact: dict) -> dict:
    return {
        "material_visit_share_snapshot": copy.deepcopy(
            default_selection_score_artifact.get(
                "first_selected_material_visit_share_snapshot"
            )
        ),
        "meaningful_q_support_snapshot": copy.deepcopy(
            default_selection_score_artifact.get(
                "first_selected_meaningful_q_support_snapshot"
            )
        ),
        "selection_score_overtake_snapshot": copy.deepcopy(
            default_selection_score_artifact.get(
                "first_selected_selection_score_overtake_snapshot"
            )
        ),
    }


def _no_first_support(first_support_summary: dict) -> bool:
    return all(value is None for value in first_support_summary.values())


def classify_decomposition(
    *,
    final_margin_summary: dict,
    first_support_summary: dict,
    relaxed_thresholds: dict,
) -> tuple[str, str]:
    relaxed_q_margin = final_margin_summary["relaxed_q_margin"]
    relaxed_selection_score_margin = final_margin_summary[
        "relaxed_selection_score_margin"
    ]
    relaxed_visit_share_margin = final_margin_summary["relaxed_visit_share_margin"]
    relaxed_visit_share_threshold = relaxed_thresholds["material_visit_share_margin"]

    positive_final_margin_count = sum(
        margin > 0
        for margin in (
            relaxed_q_margin,
            relaxed_selection_score_margin,
            relaxed_visit_share_margin,
        )
    )
    if positive_final_margin_count >= 2:
        return (
            "metric_co_movement",
            "Multiple final margins move in the selected move's direction, but none reaches material support thresholds cleanly enough to isolate a single mechanism.",
        )

    no_first_support = _no_first_support(first_support_summary)
    if (
        no_first_support
        and relaxed_visit_share_threshold - BOUNDARY_BAND
        <= relaxed_visit_share_margin
        < relaxed_visit_share_threshold
    ):
        return (
            "threshold_boundary_ambiguity",
            "Visit-share evidence sits inside the relaxed threshold boundary band without first-support snapshots.",
        )
    if (
        no_first_support
        and relaxed_q_margin <= 0
        and relaxed_selection_score_margin <= 0
        and relaxed_visit_share_margin < relaxed_visit_share_threshold - BOUNDARY_BAND
    ):
        return (
            "signal_absent",
            "Adequate cadence exists, but final Q, selection-score, and visit-share evidence do not support the selected move.",
        )
    return (
        "decomposition_inconclusive",
        "The validated chain is non-separable, but the margin and first-support pattern does not match a narrower decomposition branch.",
    )


def build_payload(
    default_selection_score_artifact: dict,
    trace_cadence_review_artifact: dict,
    threshold_review_artifact: dict,
    nonseparable_review_artifact: dict,
    *,
    source_selection_score_artifact_path: str,
    source_trace_cadence_review_artifact_path: str,
    source_threshold_review_artifact_path: str,
    source_nonseparable_review_artifact_path: str,
) -> dict:
    validate_selection_score_artifact_contract(
        default_selection_score_artifact,
        context="default selection score artifact",
        thresholds=BASELINE_THRESHOLDS,
    )
    validate_selection_score_artifact_contract(
        threshold_review_artifact,
        context="threshold review artifact",
        thresholds=RELAXED_THRESHOLDS,
    )
    validate_trace_cadence_review_artifact_contract(trace_cadence_review_artifact)
    validate_nonseparable_review_artifact_contract(nonseparable_review_artifact)
    validate_input_chain_identity(
        default_selection_score_artifact,
        trace_cadence_review_artifact,
        threshold_review_artifact,
        nonseparable_review_artifact,
        source_selection_score_artifact_path=source_selection_score_artifact_path,
        source_trace_cadence_review_artifact_path=source_trace_cadence_review_artifact_path,
        source_threshold_review_artifact_path=source_threshold_review_artifact_path,
    )

    first_support_summary = _first_support_summary(default_selection_score_artifact)
    final_margin_summary = {
        "default_q_margin": default_selection_score_artifact[
            "final_selected_minus_reference_q"
        ],
        "default_selection_score_margin": default_selection_score_artifact[
            "final_selected_minus_reference_selection_score"
        ],
        "default_visit_share_margin": default_selection_score_artifact[
            "final_selected_minus_reference_visit_share"
        ],
        "relaxed_q_margin": threshold_review_artifact[
            "final_selected_minus_reference_q"
        ],
        "relaxed_selection_score_margin": threshold_review_artifact[
            "final_selected_minus_reference_selection_score"
        ],
        "relaxed_visit_share_margin": threshold_review_artifact[
            "final_selected_minus_reference_visit_share"
        ],
    }
    classification, evidence_summary = classify_decomposition(
        final_margin_summary=final_margin_summary,
        first_support_summary=first_support_summary,
        relaxed_thresholds=threshold_review_artifact["thresholds"],
    )
    source_artifact = _validated_selection_source_artifact(
        default_selection_score_artifact,
        context="default selection score artifact",
    )
    return {
        "schema": SCHEMA,
        "hypothesis": "genuinely_not_separable_decomposition",
        "classification": {
            "classification": classification,
            "evidence_summary": evidence_summary,
        },
        "decision": CLASSIFICATION_DECISIONS[classification],
        "input_artifacts": {
            "source_selection_score_artifact_path": source_selection_score_artifact_path,
            "source_trace_cadence_review_artifact_path": source_trace_cadence_review_artifact_path,
            "source_threshold_review_artifact_path": source_threshold_review_artifact_path,
            "source_nonseparable_review_artifact_path": source_nonseparable_review_artifact_path,
        },
        "source_artifact": source_artifact,
        "thresholds_evaluated": {
            "default": copy.deepcopy(default_selection_score_artifact["thresholds"]),
            "relaxed": copy.deepcopy(threshold_review_artifact["thresholds"]),
            "threshold_boundary_band": BOUNDARY_BAND,
        },
        "final_margin_summary": final_margin_summary,
        "first_support_summary": first_support_summary,
        "cadence_summary": {
            "classification": trace_cadence_review_artifact["classification"][
                "classification"
            ],
            "decision": trace_cadence_review_artifact["decision"],
            "unique_simulation_checkpoint_count": trace_cadence_review_artifact[
                "unique_simulation_checkpoint_count"
            ],
            "unique_simulation_checkpoints": copy.deepcopy(
                trace_cadence_review_artifact["unique_simulation_checkpoints"]
            ),
        },
        "source_snapshots": {
            "default_classification": copy.deepcopy(
                default_selection_score_artifact["classification"]
            ),
            "cadence_classification": copy.deepcopy(
                trace_cadence_review_artifact["classification"]
            ),
            "threshold_classification": copy.deepcopy(
                threshold_review_artifact["classification"]
            ),
            "nonseparable_classification": copy.deepcopy(
                nonseparable_review_artifact["classification"]
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    default_selection_score_artifact = load_json(args.source_selection_score_artifact)
    threshold_review_artifact = load_json(args.source_threshold_review_artifact)
    trace_cadence_review_artifact = load_json(args.source_trace_cadence_review_artifact)
    nonseparable_review_artifact = load_json(args.source_nonseparable_review_artifact)
    payload = build_payload(
        default_selection_score_artifact,
        trace_cadence_review_artifact,
        threshold_review_artifact,
        nonseparable_review_artifact,
        source_selection_score_artifact_path=str(args.source_selection_score_artifact),
        source_trace_cadence_review_artifact_path=str(
            args.source_trace_cadence_review_artifact
        ),
        source_threshold_review_artifact_path=str(
            args.source_threshold_review_artifact
        ),
        source_nonseparable_review_artifact_path=str(
            args.source_nonseparable_review_artifact
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
