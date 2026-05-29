from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SCHEMA = "azlite_capture_002_003_guarded_w2_config_derivation_v1"
ROOT_PRIOR_SCHEMA = "azlite_capture_002_root_prior_intervention_v1"
EXPECTED_CLASSIFICATION = "policy_prior_sensitive"
EXPECTED_NEXT_BRANCH = "policy-prior calibration experiment"
DECISION = "write_guarded_w2_prior_calibration_spec"
ROW_IDS = ("capture_available-002", "capture_available-003")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive the next guarded w2 calibration config from existing diagnostics"
    )
    parser.add_argument("--root-prior-summary", type=Path, required=True)
    parser.add_argument("--guarded-w2-gate", type=Path, required=True)
    parser.add_argument("--policy-target-gate", type=Path, required=True)
    parser.add_argument("--value-target-aligned-gate", type=Path, required=True)
    parser.add_argument("--policy-target-arena", type=Path, required=True)
    parser.add_argument("--value-target-aligned-arena", type=Path, required=True)
    parser.add_argument("--guarded-w2-runtime-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_root_prior_summary(summary: dict) -> dict:
    if not isinstance(summary, dict):
        raise ValueError("root prior summary must be an object")
    if summary.get("schema") != ROOT_PRIOR_SCHEMA:
        raise ValueError("root prior summary has wrong schema")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("root prior summary must contain artifacts")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("root prior artifact entries must be objects")
        if artifact.get("classification") != EXPECTED_CLASSIFICATION:
            raise ValueError(
                "root prior summary must classify artifacts as policy_prior_sensitive"
            )
        if artifact.get("recommended_next_branch") != EXPECTED_NEXT_BRANCH:
            raise ValueError(
                "root prior summary must recommend the policy-prior calibration experiment"
            )
    return copy.deepcopy(summary)


def _validate_gate(gate: dict, *, context: str) -> dict:
    if not isinstance(gate, dict):
        raise ValueError(f"{context} gate must be an object")
    if not isinstance(gate.get("rows"), dict):
        raise ValueError(f"{context} gate must contain rows")
    for row_id in ROW_IDS:
        if row_id not in gate["rows"]:
            raise ValueError(f"{context} gate missing {row_id}")
    return copy.deepcopy(gate)


def _validate_arena(arena_report: dict, *, context: str) -> dict:
    if not isinstance(arena_report, dict):
        raise ValueError(f"{context} arena report must be an object")
    if arena_report.get("schema") != "arena_v1":
        raise ValueError(f"{context} arena report must use schema arena_v1")
    return copy.deepcopy(arena_report)


def gate_row_metrics(gate: dict, row_id: str) -> dict:
    row = gate["rows"][row_id]
    candidate = row["candidate"]
    return {
        "searched_selected_move": candidate.get("searched_selected_move"),
        "reference_move_visit_share": candidate.get("reference_move_visit_share"),
        "policy_reference_probability": candidate.get("policy_reference_probability"),
        "selected_minus_reference_q_margin": candidate.get(
            "selected_minus_reference_q_margin"
        ),
        "pass_fail_reason": row.get("pass_fail_reason"),
    }


def build_payload(
    *,
    root_prior_summary: dict,
    guarded_w2_gate: dict,
    policy_target_gate: dict,
    value_target_aligned_gate: dict,
    policy_target_arena: dict,
    value_target_aligned_arena: dict,
    guarded_w2_runtime_config: dict,
    input_paths: dict[str, str],
) -> dict:
    validated_root_prior_summary = _validate_root_prior_summary(root_prior_summary)
    validated_guarded_w2_gate = _validate_gate(guarded_w2_gate, context="guarded_w2")
    validated_policy_target_gate = _validate_gate(
        policy_target_gate, context="policy_target"
    )
    validated_value_target_aligned_gate = _validate_gate(
        value_target_aligned_gate, context="value_target_aligned"
    )
    validated_policy_target_arena = _validate_arena(
        policy_target_arena, context="policy_target"
    )
    validated_value_target_aligned_arena = _validate_arena(
        value_target_aligned_arena, context="value_target_aligned"
    )
    if not isinstance(guarded_w2_runtime_config, dict):
        raise ValueError("guarded w2 runtime config must be an object")

    guarded_002 = gate_row_metrics(validated_guarded_w2_gate, "capture_available-002")
    guarded_003 = gate_row_metrics(validated_guarded_w2_gate, "capture_available-003")
    policy_002 = gate_row_metrics(validated_policy_target_gate, "capture_available-002")
    policy_003 = gate_row_metrics(validated_policy_target_gate, "capture_available-003")
    value_002 = gate_row_metrics(
        validated_value_target_aligned_gate, "capture_available-002"
    )
    value_003 = gate_row_metrics(
        validated_value_target_aligned_gate, "capture_available-003"
    )

    recommended_runtime_config_delta = {
        "base_runtime_config": input_paths["guarded_w2_runtime_config"],
        "retain": [
            "rule-conditioned opening full guarded replay artifact",
            "replay_weight=2",
            "sharpened policy-target mode",
            "sharpened value-target mode",
            "current deterministic search-control diagnostics bundle",
        ],
        "do_not_change": [
            "guarded fixed replay source composition",
            "002/003 local gate definitions",
            "arena budgets",
            "MCTS1200 benchmark settings",
        ],
        "next_training_branch_shape": {
            "kind": "diagnostics_only_spec",
            "target": "guarded_w2_prior_calibration",
            "principle": "narrow prior-calibration change on the guarded w2 base with explicit 003 preservation constraints",
        },
        "recommended_constraints": [
            "Any training change must be validated against capture_available-002 and capture_available-003 before arena",
            "Reject broad sharpened-target changes that lower row 002 reference prior support below guarded w2",
            "Reject any branch that changes row 003 searched selected move away from reference move 1",
            "Prefer changes that raise row 002 reference prior/probability without increasing row 002 selected-minus-reference Q margin",
        ],
        "explicit_non_recommendations": [
            "Do not reuse the policy-target-local lane as the next base",
            "Do not reuse the value-target-aligned-local lane as the next base",
            "Do not broaden into generic value-target sharpening before guarded row-pair preservation is specified",
        ],
    }

    return {
        "schema": SCHEMA,
        "hypothesis": "guarded_w2_prior_calibration_derivation",
        "classification": {
            "classification": "guarded_w2_is_best_supported_base",
            "evidence_summary": (
                "Root-prior intervention classified the mechanism as policy-prior-sensitive, "
                "but both broad sharpened-target training lanes underperformed the guarded w2 base "
                "on the tracked 002/003 rows, and the value-target-aligned lane catastrophically failed arena."
            ),
        },
        "decision": DECISION,
        "input_artifacts": input_paths,
        "source_snapshot": {
            "root_prior_summary": {
                "schema": validated_root_prior_summary.get("schema"),
                "recommended_next_branch": validated_root_prior_summary.get(
                    "recommended_next_branch"
                ),
                "artifact_classifications": [
                    artifact.get("classification")
                    for artifact in validated_root_prior_summary.get("artifacts", [])
                ],
            },
            "guarded_w2_runtime_config_run_id": guarded_w2_runtime_config.get("run_id"),
        },
        "comparative_evidence": {
            "guarded_w2": {
                "row_002": guarded_002,
                "row_003": guarded_003,
            },
            "policy_target_local": {
                "arena_score": validated_policy_target_arena.get("score"),
                "row_002": policy_002,
                "row_003": policy_003,
            },
            "value_target_aligned_local": {
                "arena_score": validated_value_target_aligned_arena.get("score"),
                "row_002": value_002,
                "row_003": value_003,
            },
        },
        "recommended_runtime_config_delta": recommended_runtime_config_delta,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(
        root_prior_summary=load_json(args.root_prior_summary),
        guarded_w2_gate=load_json(args.guarded_w2_gate),
        policy_target_gate=load_json(args.policy_target_gate),
        value_target_aligned_gate=load_json(args.value_target_aligned_gate),
        policy_target_arena=load_json(args.policy_target_arena),
        value_target_aligned_arena=load_json(args.value_target_aligned_arena),
        guarded_w2_runtime_config=load_json(args.guarded_w2_runtime_config),
        input_paths={
            "root_prior_summary": str(args.root_prior_summary),
            "guarded_w2_gate": str(args.guarded_w2_gate),
            "policy_target_gate": str(args.policy_target_gate),
            "value_target_aligned_gate": str(args.value_target_aligned_gate),
            "policy_target_arena": str(args.policy_target_arena),
            "value_target_aligned_arena": str(args.value_target_aligned_arena),
            "guarded_w2_runtime_config": str(args.guarded_w2_runtime_config),
        },
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
