from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SCHEMA = "azlite_capture_002_non_residual_mechanism_review_v1"
SOURCE_SELECTION_PRESSURE_ABLATION_SCHEMA = (
    "azlite_capture_002_selection_pressure_ablation_v1"
)
ROW_ID = "capture_available-002"
SOURCE_CLASSIFICATION_DECISIONS = {
    "selection_pressure_persists": "stop_002_selection_pressure_ablation_inconclusive",
    "selection_pressure_variant_sensitive": "write_002_search_pressure_variant_followup_spec",
}
CLASSIFICATION_DECISIONS = {
    "stable_non_residual_selection_advantage": "write_002_non_residual_mechanism_review_spec",
    "search_option_sensitive": "write_002_search_pressure_variant_followup_spec",
    "review_prerequisite_blocked": "stop_002_non_residual_mechanism_review_inconclusive",
}
PRESSURE_DECISION = "write_002_selection_pressure_ablation_spec"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review whether capture 002 pressure persists beyond tested search-pressure variants"
    )
    parser.add_argument(
        "--source-selection-pressure-ablation-artifact", type=Path, required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validated_variant(variant: dict, *, index: int) -> dict:
    if not isinstance(variant, dict):
        raise ValueError(f"variants[{index}] must be an object")
    if not isinstance(variant.get("variant"), str) or not variant.get("variant"):
        raise ValueError(f"variants[{index}].variant must be a non-empty string")
    if not isinstance(variant.get("label"), str) or not variant.get("label"):
        raise ValueError(f"variants[{index}].label must be a non-empty string")
    if not isinstance(variant.get("pressure_relieved"), bool):
        raise ValueError(f"variants[{index}].pressure_relieved must be boolean")
    return variant


def _validated_artifact(artifact: dict) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("selection-pressure ablation artifact must be an object")
    if artifact.get("schema") != SOURCE_SELECTION_PRESSURE_ABLATION_SCHEMA:
        raise ValueError("selection-pressure ablation artifact has wrong schema")
    if artifact.get("row_id") != ROW_ID:
        raise ValueError(
            f"selection-pressure ablation artifact row_id must be {ROW_ID}"
        )
    classification = artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError(
            "selection-pressure ablation artifact classification must be an object"
        )
    classification_name = classification.get("classification")
    expected_decision = SOURCE_CLASSIFICATION_DECISIONS.get(classification_name)
    decision = artifact.get("decision")
    if not isinstance(decision, str) or not decision:
        raise ValueError(
            "selection-pressure ablation artifact decision must be a non-empty string"
        )
    if expected_decision is None or decision != expected_decision:
        raise ValueError(
            "selection-pressure ablation artifact must use a supported classification/decision pair"
        )
    variants = artifact.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(
            "selection-pressure ablation artifact variants must be a non-empty list"
        )
    validated_variants = [
        _validated_variant(variant, index=index)
        for index, variant in enumerate(variants)
    ]
    return {
        **artifact,
        "classification": copy.deepcopy(classification),
        "variants": validated_variants,
    }


def build_payload(
    selection_pressure_ablation_artifact: dict,
    *,
    source_selection_pressure_ablation_artifact_path: str,
) -> dict:
    artifact = _validated_artifact(selection_pressure_ablation_artifact)
    variants = artifact["variants"]
    relieved_variants = [
        variant for variant in variants if variant["pressure_relieved"]
    ]
    downstream_valid_variants = [
        variant
        for variant in variants
        if isinstance(variant.get("default_trace"), dict)
        and isinstance(variant.get("relaxed_trace"), dict)
    ]
    persistent_variants = [
        variant
        for variant in downstream_valid_variants
        if variant.get("default_trace", {}).get("decision") == PRESSURE_DECISION
        and variant.get("relaxed_trace", {}).get("decision") == PRESSURE_DECISION
    ]

    if relieved_variants:
        classification_name = "search_option_sensitive"
        evidence_summary = "At least one tested search variant relieved the early selection-score lead."
    elif persistent_variants:
        classification_name = "stable_non_residual_selection_advantage"
        evidence_summary = (
            "Every downstream-valid tested variant preserved the early selection-score lead;"
            " the observed mechanism is not explained by the tested search-pressure settings."
        )
    else:
        classification_name = "review_prerequisite_blocked"
        evidence_summary = "No downstream-valid variant remained available for a non-residual mechanism review."

    return {
        "schema": SCHEMA,
        "hypothesis": "non_residual_mechanism_review",
        "classification": {
            "classification": classification_name,
            "evidence_summary": evidence_summary,
        },
        "decision": CLASSIFICATION_DECISIONS[classification_name],
        "input_artifacts": {
            "source_selection_pressure_ablation_artifact_path": source_selection_pressure_ablation_artifact_path,
        },
        "row_id": ROW_ID,
        "candidate_path": artifact.get("candidate_path"),
        "variant_summary": {
            "variant_count": len(variants),
            "downstream_valid_variant_count": len(downstream_valid_variants),
            "persistent_variant_count": len(persistent_variants),
            "relieved_variant_count": len(relieved_variants),
            "persistent_variants": [
                variant["variant"] for variant in persistent_variants
            ],
            "relieved_variants": [variant["variant"] for variant in relieved_variants],
        },
        "source_snapshot": {
            "selection_pressure_ablation_schema": artifact.get("schema"),
            "selection_pressure_ablation_classification": copy.deepcopy(
                artifact.get("classification")
            ),
            "selection_pressure_ablation_decision": artifact.get("decision"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selection_pressure_ablation_artifact = load_json(
        args.source_selection_pressure_ablation_artifact
    )
    payload = build_payload(
        selection_pressure_ablation_artifact,
        source_selection_pressure_ablation_artifact_path=str(
            args.source_selection_pressure_ablation_artifact
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
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
