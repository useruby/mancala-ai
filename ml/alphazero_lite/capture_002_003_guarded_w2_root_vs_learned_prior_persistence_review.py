from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SCHEMA = (
    "azlite_capture_002_003_guarded_w2_root_vs_learned_prior_persistence_review_v1"
)
SOURCE_ROOT_PRIOR_SCHEMA = "azlite_capture_002_root_prior_intervention_v1"
EXPECTED_ROOT_CLASSIFICATION = "policy_prior_sensitive"
CLASSIFICATION_DECISIONS = {
    "root_override_effective_but_training_nonpersistent": "write_guarded_w2_root_vs_learned_prior_persistence_spec",
    "review_prerequisite_blocked": "stop_guarded_w2_root_vs_learned_prior_persistence_review_inconclusive",
}
ROW_002 = "capture_available-002"
ROW_003 = "capture_available-003"
REQUIRED_BUDGETS = (384, 1200)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review whether guarded w2 root-prior success persists through learned prior calibration"
    )
    parser.add_argument("--root-prior-summary", type=Path, required=True)
    parser.add_argument("--guarded-w2-gate", type=Path, required=True)
    parser.add_argument("--prior-calibration-gate", type=Path, required=True)
    parser.add_argument("--prior-calibration-arena", type=Path, required=True)
    parser.add_argument("--guarded-w2-train-log", type=Path, required=True)
    parser.add_argument("--prior-calibration-train-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_best_val_loss(log_text: str, *, context: str) -> float:
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if line.startswith("best_val_loss="):
            return float(line.split("=", 1)[1])
    raise ValueError(f"{context} must contain best_val_loss=")


def _validate_gate(gate: dict, *, context: str) -> dict:
    if not isinstance(gate, dict):
        raise ValueError(f"{context} gate must be an object")
    if gate.get("schema") != "azlite_rule_conditioned_opening_local_gate_v1":
        raise ValueError(f"{context} gate has wrong schema")
    rows = gate.get("rows")
    if not isinstance(rows, dict):
        raise ValueError(f"{context} gate must contain rows")
    for row_id in (ROW_002, ROW_003):
        if row_id not in rows:
            raise ValueError(f"{context} gate missing {row_id}")
    return copy.deepcopy(gate)


def _validate_arena(report: dict) -> dict:
    if not isinstance(report, dict):
        raise ValueError("arena report must be an object")
    if report.get("schema") != "arena_v1":
        raise ValueError("arena report has wrong schema")
    return copy.deepcopy(report)


def _guarded_root_prior_artifact(summary: dict, *, guarded_candidate_path: str) -> dict:
    if not isinstance(summary, dict):
        raise ValueError("root prior summary must be an object")
    if summary.get("schema") != SOURCE_ROOT_PRIOR_SCHEMA:
        raise ValueError("root prior summary has wrong schema")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("root prior summary artifacts must be a list")
    for artifact in artifacts:
        if (
            isinstance(artifact, dict)
            and artifact.get("artifact_path") == guarded_candidate_path
        ):
            if artifact.get("classification") != EXPECTED_ROOT_CLASSIFICATION:
                raise ValueError("guarded w2 root prior artifact must be policy_prior_sensitive")
            return copy.deepcopy(artifact)
    raise ValueError("guarded w2 root prior artifact not found in summary")


def _result_index(results: list[dict]) -> dict[tuple[str, str, int], dict]:
    index: dict[tuple[str, str, int], dict] = {}
    for result in results:
        key = (
            str(result.get("intervention")),
            str(result.get("row_id")),
            int(result.get("simulations")),
        )
        index[key] = result
    return index


def _row002_root_success(result: dict | None) -> bool:
    return bool(result and result.get("selected_is_reference") is True)


def _row003_root_preserved(result: dict | None) -> bool:
    return bool(
        result
        and (
            result.get("pass_fail_reason") == "pass_reference_preserved"
            or result.get("selected_is_reference") is True
        )
    )


def _intervention_summary(index: dict[tuple[str, str, int], dict], intervention: str) -> dict:
    row_002_selected_budgets = []
    row_003_preserved_budgets = []
    for budget in (32, 64, 128, 384, 1200):
        row_002_result = index.get((intervention, ROW_002, budget))
        row_003_result = index.get((intervention, ROW_003, budget))
        if _row002_root_success(row_002_result):
            row_002_selected_budgets.append(budget)
        if _row003_root_preserved(row_003_result):
            row_003_preserved_budgets.append(budget)
    return {
        "intervention": intervention,
        "row_002_selected_budgets": row_002_selected_budgets,
        "row_003_preserved_budgets": row_003_preserved_budgets,
        "persistent_at_required_budgets": all(
            budget in row_002_selected_budgets and budget in row_003_preserved_budgets
            for budget in REQUIRED_BUDGETS
        ),
        "first_row_002_selected_budget": (
            row_002_selected_budgets[0] if row_002_selected_budgets else None
        ),
    }


def _gate_metrics(gate: dict, row_id: str) -> dict:
    candidate = gate["rows"][row_id]["candidate"]
    return {
        "policy_reference_probability": candidate.get("policy_reference_probability"),
        "reference_move_visit_share": candidate.get("reference_move_visit_share"),
        "searched_selected_move": candidate.get("searched_selected_move"),
        "selected_minus_reference_q_margin": candidate.get(
            "selected_minus_reference_q_margin"
        ),
        "pass_fail_reason": gate["rows"][row_id].get("pass_fail_reason"),
    }


def build_payload(
    root_prior_summary: dict,
    guarded_w2_gate: dict,
    prior_calibration_gate: dict,
    prior_calibration_arena: dict,
    guarded_w2_train_log: str,
    prior_calibration_train_log: str,
    *,
    source_root_prior_summary_path: str,
    source_guarded_w2_gate_path: str,
    source_prior_calibration_gate_path: str,
    source_prior_calibration_arena_path: str,
    source_guarded_w2_train_log_path: str,
    source_prior_calibration_train_log_path: str,
) -> dict:
    validated_guarded_gate = _validate_gate(guarded_w2_gate, context="guarded_w2")
    validated_prior_gate = _validate_gate(
        prior_calibration_gate, context="prior_calibration"
    )
    validated_arena = _validate_arena(prior_calibration_arena)
    guarded_root_artifact = _guarded_root_prior_artifact(
        root_prior_summary,
        guarded_candidate_path=str(validated_guarded_gate.get("candidate_path")),
    )

    guarded_train_best_val_loss = _parse_best_val_loss(
        guarded_w2_train_log, context="guarded_w2 train log"
    )
    prior_calibration_train_best_val_loss = _parse_best_val_loss(
        prior_calibration_train_log, context="prior_calibration train log"
    )

    root_index = _result_index(list(guarded_root_artifact.get("results") or []))
    root_intervention_summaries = [
        _intervention_summary(root_index, intervention)
        for intervention in (
            "uniform_legal_prior",
            "zero_wrong_extra_turn_prior",
            "equalize_reference_and_wrong",
            "swap_reference_and_wrong",
            "force_reference_prior_advantage",
        )
    ]
    persistent_root_interventions = [
        summary["intervention"]
        for summary in root_intervention_summaries
        if summary["persistent_at_required_budgets"]
    ]

    guarded_002 = _gate_metrics(validated_guarded_gate, ROW_002)
    guarded_003 = _gate_metrics(validated_guarded_gate, ROW_003)
    learned_002 = _gate_metrics(validated_prior_gate, ROW_002)
    learned_003 = _gate_metrics(validated_prior_gate, ROW_003)

    row_002_prior_improved = float(learned_002["policy_reference_probability"]) > float(
        guarded_002["policy_reference_probability"]
    )
    row_002_visit_share_improved = float(learned_002["reference_move_visit_share"]) > float(
        guarded_002["reference_move_visit_share"]
    )
    row_002_selection_still_not_fixed = int(learned_002["searched_selected_move"]) != 4
    row_002_q_margin_worsened = float(learned_002["selected_minus_reference_q_margin"]) > float(
        guarded_002["selected_minus_reference_q_margin"]
    )
    arena_score = float(validated_arena.get("score", 0.0))

    if (
        persistent_root_interventions
        and row_002_prior_improved
        and row_002_visit_share_improved
        and row_002_selection_still_not_fixed
    ):
        classification_name = "root_override_effective_but_training_nonpersistent"
        evidence_summary = (
            "Guarded w2 root-only prior interventions can sustain row 002 reference selection "
            "while preserving row 003 at high budgets, but the learned prior-calibration retry "
            "only increased row 002 prior support without holding the flip through final search selection, "
            "and downstream arena collapsed."
        )
    else:
        classification_name = "review_prerequisite_blocked"
        evidence_summary = (
            "The guarded w2 root-prior evidence and learned retry evidence did not support a clean persistence review conclusion."
        )

    return {
        "schema": SCHEMA,
        "hypothesis": "guarded_w2_root_vs_learned_prior_persistence",
        "classification": {
            "classification": classification_name,
            "evidence_summary": evidence_summary,
        },
        "decision": CLASSIFICATION_DECISIONS[classification_name],
        "input_artifacts": {
            "source_root_prior_summary_path": source_root_prior_summary_path,
            "source_guarded_w2_gate_path": source_guarded_w2_gate_path,
            "source_prior_calibration_gate_path": source_prior_calibration_gate_path,
            "source_prior_calibration_arena_path": source_prior_calibration_arena_path,
            "source_guarded_w2_train_log_path": source_guarded_w2_train_log_path,
            "source_prior_calibration_train_log_path": source_prior_calibration_train_log_path,
        },
        "source_snapshot": {
            "guarded_w2_root_prior_artifact": {
                "artifact": guarded_root_artifact.get("artifact"),
                "artifact_path": guarded_root_artifact.get("artifact_path"),
                "classification": guarded_root_artifact.get("classification"),
                "recommended_next_branch": guarded_root_artifact.get(
                    "recommended_next_branch"
                ),
            },
            "guarded_w2_train_best_val_loss": guarded_train_best_val_loss,
            "prior_calibration_train_best_val_loss": prior_calibration_train_best_val_loss,
        },
        "root_override_review": {
            "required_budgets": list(REQUIRED_BUDGETS),
            "interventions": root_intervention_summaries,
            "persistent_root_interventions": persistent_root_interventions,
        },
        "learned_retry_review": {
            "guarded_w2": {
                "row_002": guarded_002,
                "row_003": guarded_003,
            },
            "prior_calibration": {
                "row_002": learned_002,
                "row_003": learned_003,
                "arena_score": arena_score,
            },
            "comparisons": {
                "row_002_prior_improved": row_002_prior_improved,
                "row_002_visit_share_improved": row_002_visit_share_improved,
                "row_002_selection_still_not_fixed": row_002_selection_still_not_fixed,
                "row_002_q_margin_worsened": row_002_q_margin_worsened,
                "arena_collapsed": arena_score == 0.0,
                "best_val_loss_delta": round(
                    prior_calibration_train_best_val_loss - guarded_train_best_val_loss,
                    6,
                ),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(
        load_json(args.root_prior_summary),
        load_json(args.guarded_w2_gate),
        load_json(args.prior_calibration_gate),
        load_json(args.prior_calibration_arena),
        _load_text(args.guarded_w2_train_log),
        _load_text(args.prior_calibration_train_log),
        source_root_prior_summary_path=str(args.root_prior_summary),
        source_guarded_w2_gate_path=str(args.guarded_w2_gate),
        source_prior_calibration_gate_path=str(args.prior_calibration_gate),
        source_prior_calibration_arena_path=str(args.prior_calibration_arena),
        source_guarded_w2_train_log_path=str(args.guarded_w2_train_log),
        source_prior_calibration_train_log_path=str(args.prior_calibration_train_log),
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
