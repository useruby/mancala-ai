#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    AuditRow,
    CHILD_PUCT_BUDGETS,
    CHILD_TEACHER_BASE_BUDGETS,
    CHILD_TEACHER_SEEDS,
    C_PUCT,
    FAMILY,
    OPTIONAL_CHILD_TEACHER_BUDGET,
    ROOT_BUDGETS,
    child_neural_audit,
    child_puct_audit,
    classify_row,
    counterfactual_rows,
    load_json,
    load_reference_maps,
    python_bin,
    repo_root,
    root_baseline_for_row,
    round_float,
    teacher_child_audit,
    write_json,
)


DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_MINING_SUMMARY_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining/non_opening_failure_family_summary.json"
)
DEFAULT_SELECTED_ROWS_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining/selected_non_opening_family_rows.jsonl"
)
DEFAULT_OLD_AUDIT_SUMMARY_PATH = Path(
    "/tmp/azlite_incumbent_proxy_value_backup_audit/"
    "incumbent_proxy_value_backup_audit_summary.json"
)
DEFAULT_SUMMARY_OUT = Path(
    "/tmp/azlite_incumbent_proxy_post_adjudication_rebaseline/"
    "incumbent_proxy_post_adjudication_rebaseline_summary.json"
)
DEFAULT_REPORT_OUT = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-post-adjudication-rebaseline-results.md"
)
SCHEMA = "azlite_incumbent_proxy_disagreement_post_adjudication_rebaseline_v1"
OLD_REFERENCE_MOVE_021 = 3
EXPECTED_REFERENCE_MOVE_021 = 2
OLD_AUDITED_ROW_IDS = {
    "incumbent_proxy_disagreement-007",
    "incumbent_proxy_disagreement-008",
    "incumbent_proxy_disagreement-009",
    "incumbent_proxy_disagreement-011",
    "incumbent_proxy_disagreement-014",
    "incumbent_proxy_disagreement-019",
    "incumbent_proxy_disagreement-021",
    "incumbent_proxy_disagreement-022",
    "incumbent_proxy_disagreement-023",
    "incumbent_proxy_disagreement-024",
    "incumbent_proxy_disagreement-025",
    "incumbent_proxy_disagreement-026",
    "incumbent_proxy_disagreement-028",
    "incumbent_proxy_disagreement-029",
    "incumbent_proxy_disagreement-030",
    "incumbent_proxy_disagreement-035",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument(
        "--mining-summary-path", type=Path, default=DEFAULT_MINING_SUMMARY_PATH
    )
    parser.add_argument(
        "--selected-rows-path", type=Path, default=DEFAULT_SELECTED_ROWS_PATH
    )
    parser.add_argument(
        "--old-audit-summary-path", type=Path, default=DEFAULT_OLD_AUDIT_SUMMARY_PATH
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpuct", type=float, default=C_PUCT)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(payload)
    return rows


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return output


def root_failure_mode_from_baseline(baseline: dict[str, Any]) -> str:
    if bool(baseline["pass"]):
        return "pass"
    reference_q = baseline.get("reference_q")
    selected_q = baseline.get("selected_q")
    reference_u = baseline.get("reference_u")
    selected_u = baseline.get("selected_u")
    reference_prior = baseline.get("reference_prior")
    selected_prior = baseline.get("selected_prior")
    if reference_q is not None and selected_q is not None and selected_q > reference_q:
        if (
            reference_u is not None
            and selected_u is not None
            and selected_u > reference_u
        ):
            return "value_q"
        return "value_q"
    if reference_u is not None and selected_u is not None and selected_u > reference_u:
        return "search_selection"
    if (
        reference_prior is not None
        and selected_prior is not None
        and selected_prior > reference_prior
    ):
        return "policy_only"
    return "unknown"


def prior_rank(root_policy: dict[str, Any], move: int) -> int | None:
    pairs: list[tuple[int, float]] = []
    for key, value in root_policy.items():
        pairs.append((int(key), float(value)))
    if not pairs:
        return None
    ordered = sorted(pairs, key=lambda item: (-item[1], item[0]))
    for index, (candidate_move, _value) in enumerate(ordered, start=1):
        if candidate_move == int(move):
            return index
    return None


def current_status_from_baseline(baseline: dict[str, Any]) -> str:
    return (
        "pass_corrected_reference"
        if bool(baseline["pass"])
        else "fail_corrected_reference"
    )


def rerun_family_mining(root: Path) -> None:
    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.run_corrected_non_opening_failure_family_mining",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"family mining refresh failed with exit code {completed.returncode}"
        )


def fixture_verification(
    *,
    suite_by_id: dict[str, Any],
    reference_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_id = "incumbent_proxy_disagreement-021"
    suite_row = suite_by_id.get(row_id)
    reference_row = reference_by_id.get(row_id)
    if suite_row is None or reference_row is None:
        status = "reference_fixture_not_updated_correctly"
        verification = {
            "row_id": row_id,
            "active_reference_move": None
            if reference_row is None
            else reference_row.get("reference_move"),
            "expected_reference_move": EXPECTED_REFERENCE_MOVE_021,
            "observed_reference_moves": None
            if reference_row is None
            else reference_row.get("observed_reference_moves"),
            "reference_unstable": None
            if reference_row is None
            else reference_row.get("reference_unstable"),
            "canonical_state_match": False,
            "status": status,
            "notes": "row missing from suite or reference fixture",
        }
        return [verification], {"ok": False, "error": status}
    suite_canonical = canonical_state_key(suite_row.state)
    reference_canonical = str(
        reference_row.get("canonical_state")
        or canonical_state_key(reference_row["state"])
    )
    canonical_state_match = suite_canonical == reference_canonical
    observed_reference_moves = list(reference_row.get("observed_reference_moves") or [])
    raw_active_reference_move = reference_row.get("reference_move")
    if raw_active_reference_move is None:
        verification = {
            "row_id": row_id,
            "active_reference_move": None,
            "expected_reference_move": EXPECTED_REFERENCE_MOVE_021,
            "observed_reference_moves": observed_reference_moves,
            "reference_unstable": reference_row.get("reference_unstable"),
            "canonical_state_match": canonical_state_match,
            "status": "reference_fixture_not_updated_correctly",
            "notes": "reference row is missing reference_move",
        }
        return [verification], {
            "ok": False,
            "error": "reference_fixture_not_updated_correctly",
        }
    active_reference_move = int(raw_active_reference_move)
    reference_unstable = bool(reference_row.get("reference_unstable", False))
    observed_ok = observed_reference_moves == [EXPECTED_REFERENCE_MOVE_021] or (
        len(observed_reference_moves) >= 1
        and set(int(move) for move in observed_reference_moves)
        == {EXPECTED_REFERENCE_MOVE_021}
    )
    ok = (
        active_reference_move == EXPECTED_REFERENCE_MOVE_021
        and not reference_unstable
        and canonical_state_match
        and observed_ok
    )
    notes = []
    if active_reference_move == EXPECTED_REFERENCE_MOVE_021:
        notes.append("active fixture row now points at move 2")
    else:
        notes.append("active fixture row does not point at move 2")
    if active_reference_move != OLD_REFERENCE_MOVE_021:
        notes.append("no active stale reference_move 3 remains for row 021")
    if observed_ok:
        notes.append("observed reference moves are consistent with the approved patch")
    else:
        notes.append(
            "observed reference moves are inconsistent with the approved patch"
        )
    verification = {
        "row_id": row_id,
        "active_reference_move": active_reference_move,
        "expected_reference_move": EXPECTED_REFERENCE_MOVE_021,
        "observed_reference_moves": observed_reference_moves,
        "reference_unstable": reference_unstable,
        "canonical_state_match": canonical_state_match,
        "status": "ok" if ok else "reference_fixture_not_updated_correctly",
        "notes": "; ".join(notes),
    }
    return [verification], {
        "ok": ok,
        "error": None if ok else "reference_fixture_not_updated_correctly",
    }


def load_family_context(
    *,
    mining_summary: dict[str, Any],
    selected_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, dict[str, Any]],
]:
    inventory_rows = [
        row
        for row in list(mining_summary.get("inventory_rows") or [])
        if isinstance(row, dict) and str(row.get("family")) == FAMILY
    ]
    inventory_rows.sort(key=lambda row: str(row["row_id"]))
    selected_by_id = {
        str(row["row_id"]): row
        for row in selected_rows
        if isinstance(row, dict) and str(row.get("family")) == FAMILY
    }
    role_by_id: dict[str, str] = {}
    for row in inventory_rows:
        row_id = str(row["row_id"])
        selected_row = selected_by_id.get(row_id)
        if selected_row is not None:
            role_by_id[row_id] = str(selected_row["recommended_role"])
        elif str(row.get("failure_status")) == "fail_corrected_reference":
            role_by_id[row_id] = "target_candidate"
        else:
            role_by_id[row_id] = "pass"
    representative_rows = {
        str(row["row_id"]): row
        for row in list(mining_summary.get("representative_rows") or [])
        if isinstance(row, dict)
        and row.get("row_id") is not None
        and str(row.get("family")) == FAMILY
    }
    return inventory_rows, selected_by_id, role_by_id, representative_rows


def make_audit_rows(
    *,
    inventory_rows: list[dict[str, Any]],
    suite_by_id: dict[str, Any],
    reference_by_id: dict[str, dict[str, Any]],
    role_by_id: dict[str, str],
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for inventory_row in inventory_rows:
        row_id = str(inventory_row["row_id"])
        suite_row = suite_by_id[row_id]
        reference_row = reference_by_id[row_id]
        rows.append(
            AuditRow(
                row_id=row_id,
                role=role_by_id[row_id],
                severity=str(inventory_row.get("severity") or "none"),
                failure_mode=str(inventory_row.get("failure_mode") or "unknown"),
                corrected_reference_move=int(reference_row["reference_move"]),
                current_selected_move=int(
                    inventory_row.get("current_selected_move") or -1
                ),
                suite_state=dict(suite_row.state),
                legal_moves=list(suite_row.legal_moves),
                canonical_state_hash=canonical_state_key(suite_row.state),
                reference_teacher_value=round_float(reference_row.get("teacher_value")),
                reference_unstable=bool(reference_row.get("reference_unstable", False)),
                reference_integrity_error=False,
                inventory_failure_status=str(
                    inventory_row.get("failure_status") or "unknown"
                ),
                mining_metrics=dict(inventory_row),
            )
        )
    return rows


def family_baseline_table(
    *,
    audit_rows: list[AuditRow],
    evaluator: ArtifactEvaluator,
    seed: int,
    cpuct: float,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    baselines: dict[tuple[str, int], dict[str, Any]] = {}
    table: list[dict[str, Any]] = []
    summary_by_row: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        row_baselines = {}
        for budget in ROOT_BUDGETS:
            row_baseline = root_baseline_for_row(
                evaluator=evaluator,
                row=row,
                budget=int(budget),
                seed=int(seed),
                cpuct=float(cpuct),
            )
            baselines[(row.row_id, int(budget))] = row_baseline
            row_baselines[int(budget)] = row_baseline
        baseline_384 = row_baselines[384]
        baseline_1200 = row_baselines[1200]
        summary = {
            "row_id": row.row_id,
            "role": row.role,
            "corrected_reference_move": row.corrected_reference_move,
            "selected_384": baseline_384["selected_move"],
            "selected_1200": baseline_1200["selected_move"],
            "pass_384": bool(baseline_384["pass"]),
            "pass_1200": bool(baseline_1200["pass"]),
            "reference_visit_share_384": baseline_384["reference_visit_share"],
            "reference_visit_share_1200": baseline_1200["reference_visit_share"],
            "selected_minus_reference_q_margin_384": baseline_384[
                "selected_minus_reference_q_margin"
            ],
            "selected_minus_reference_q_margin_1200": baseline_1200[
                "selected_minus_reference_q_margin"
            ],
            "reference_policy_probability": baseline_1200["reference_prior"],
            "selected_policy_probability": baseline_1200["selected_prior"],
            "reference_prior_rank": prior_rank(
                baseline_1200["root_policy"], row.corrected_reference_move
            ),
            "selected_prior_rank": None
            if baseline_1200["selected_move"] is None
            else prior_rank(
                baseline_1200["root_policy"], int(baseline_1200["selected_move"])
            ),
            "pass_fail": current_status_from_baseline(baseline_1200),
            "severity": row.severity,
            "notes": f"role={row.role}; failure_mode={root_failure_mode_from_baseline(baseline_1200)}",
        }
        summary_by_row[row.row_id] = summary
        table.append(summary)
    table.sort(key=lambda row: row["row_id"])
    return table, baselines, summary_by_row


def row_021_comparison(
    *,
    row_021: AuditRow,
    evaluator: ArtifactEvaluator,
    seed: int,
    cpuct: float,
    selected_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    old_row = replace(row_021, corrected_reference_move=OLD_REFERENCE_MOVE_021)
    old_baseline_384 = root_baseline_for_row(
        evaluator=evaluator,
        row=old_row,
        budget=384,
        seed=int(seed),
        cpuct=float(cpuct),
    )
    old_baseline_1200 = root_baseline_for_row(
        evaluator=evaluator,
        row=old_row,
        budget=1200,
        seed=int(seed),
        cpuct=float(cpuct),
    )
    new_baseline_384 = root_baseline_for_row(
        evaluator=evaluator,
        row=row_021,
        budget=384,
        seed=int(seed),
        cpuct=float(cpuct),
    )
    new_baseline_1200 = root_baseline_for_row(
        evaluator=evaluator,
        row=row_021,
        budget=1200,
        seed=int(seed),
        cpuct=float(cpuct),
    )
    selected_row = selected_by_id.get(row_021.row_id)
    if selected_row is not None:
        recommended_role = str(selected_row["recommended_role"])
    elif bool(new_baseline_1200["pass"]):
        recommended_role = "pass/control"
    else:
        recommended_role = "excluded"
    return {
        "row_id": row_021.row_id,
        "old_reference_move": OLD_REFERENCE_MOVE_021,
        "new_reference_move": row_021.corrected_reference_move,
        "current_selected_384": new_baseline_384["selected_move"],
        "current_selected_1200": new_baseline_1200["selected_move"],
        "old_label_status": current_status_from_baseline(old_baseline_1200),
        "new_label_status": current_status_from_baseline(new_baseline_1200),
        "recommended_role": recommended_role,
        "old_result": {
            "selected_384": old_baseline_384["selected_move"],
            "selected_1200": old_baseline_1200["selected_move"],
            "pass_384": bool(old_baseline_384["pass"]),
            "pass_1200": bool(old_baseline_1200["pass"]),
        },
        "new_result": {
            "selected_384": new_baseline_384["selected_move"],
            "selected_1200": new_baseline_1200["selected_move"],
            "pass_384": bool(new_baseline_384["pass"]),
            "pass_1200": bool(new_baseline_1200["pass"]),
        },
        "notes": (
            "row 021 remains a corrected-reference failure after the approved move-2 update"
            if not bool(new_baseline_1200["pass"])
            else "row 021 now passes under the approved move-2 update"
        ),
    }


def audited_row_ids(
    *,
    family_baselines: list[dict[str, Any]],
    role_by_id: dict[str, str],
    old_audit_summary_path: Path,
) -> list[str]:
    row_ids = set(OLD_AUDITED_ROW_IDS)
    row_ids.update(
        row["row_id"] for row in family_baselines if not bool(row["pass_1200"])
    )
    row_ids.update(
        row_id for row_id, role in role_by_id.items() if role == "preservation_control"
    )
    if old_audit_summary_path.exists():
        old_summary = load_json(old_audit_summary_path)
        row_ids.update(
            str(row["row_id"])
            for row in list(old_summary.get("classification_table") or [])
            if isinstance(row, dict)
            and str(row.get("row_classification")) == "corrected_reference_suspicious"
        )
    return sorted(row_ids)


def run_post_adjudication_audit(
    *,
    audit_rows_by_id: dict[str, AuditRow],
    baselines: dict[tuple[str, int], dict[str, Any]],
    evaluator: ArtifactEvaluator,
    audited_ids: list[str],
    seed: int,
    cpuct: float,
) -> dict[str, Any]:
    teacher_budgets = list(CHILD_TEACHER_BASE_BUDGETS) + [OPTIONAL_CHILD_TEACHER_BUDGET]
    mechanism_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    child_neural_rows: list[dict[str, Any]] = []
    child_teacher_rows: list[dict[str, Any]] = []
    child_puct_rows: list[dict[str, Any]] = []
    counterfactual_table: list[dict[str, Any]] = []
    bucket_by_row_id: dict[str, str] = {}
    suspicious_rows: list[str] = []

    for row_id in audited_ids:
        row = audit_rows_by_id[row_id]
        baseline_1200 = baselines[(row_id, 1200)]
        if bool(baseline_1200["pass"]) and row.role == "preservation_control":
            mechanism_rows.append(
                {
                    "row_id": row_id,
                    "selected_move": baseline_1200["selected_move"],
                    "corrected_reference_move": row.corrected_reference_move,
                    "row_classification": "controls_stable",
                    "teacher_child_prefers_reference": None,
                    "neural_child_prefers_reference": None,
                    "child_puct_prefers_reference": None,
                    "root_puct_prefers_reference": True,
                    "reference_suspicious": False,
                    "recommended_use": "preserve_control",
                    "notes": "preservation control still passes at 1200",
                }
            )
            bucket_by_row_id[row_id] = "controls_stable"
            continue
        if bool(baseline_1200["pass"]):
            recommended_use = "holdout" if row.role == "holdout_candidate" else "pass"
            mechanism_rows.append(
                {
                    "row_id": row_id,
                    "selected_move": baseline_1200["selected_move"],
                    "corrected_reference_move": row.corrected_reference_move,
                    "row_classification": "pass_after_021_update",
                    "teacher_child_prefers_reference": None,
                    "neural_child_prefers_reference": None,
                    "child_puct_prefers_reference": None,
                    "root_puct_prefers_reference": True,
                    "reference_suspicious": False,
                    "recommended_use": recommended_use,
                    "notes": "row passes under the corrected post-adjudication reference",
                }
            )
            bucket_by_row_id[row_id] = "pass_after_021_update"
            continue

        neural_rows, neural_summary = child_neural_audit(
            evaluator=evaluator, row=row, baseline_1200=baseline_1200
        )
        child_neural_rows.extend(neural_rows)
        teacher_rows, teacher_budget_summary = teacher_child_audit(
            row=row,
            child_summary=neural_summary,
            budgets=tuple(teacher_budgets),
            seeds=CHILD_TEACHER_SEEDS,
        )
        child_teacher_rows.extend(teacher_rows)
        puct_rows, puct_budget_summary = child_puct_audit(
            evaluator=evaluator,
            row=row,
            child_summary=neural_summary,
            budgets=CHILD_PUCT_BUDGETS,
            seed=int(seed),
            cpuct=float(cpuct),
        )
        child_puct_rows.extend(puct_rows)
        row_counterfactuals = counterfactual_rows(
            evaluator=evaluator,
            row=row,
            child_summary=neural_summary,
            teacher_budget_summary=teacher_budget_summary,
            neural_summary=neural_summary,
            seed=int(seed),
            cpuct=float(cpuct),
        )
        counterfactual_table.extend(row_counterfactuals)
        classification = classify_row(
            row=row,
            neural_summary=neural_summary,
            teacher_budget_summary=teacher_budget_summary,
            child_puct_budget_summary=puct_budget_summary,
            root_baseline_1200=baseline_1200,
            counterfactual=row_counterfactuals,
        )
        classification_rows.append(classification)

        deepest_teacher_budget = max(teacher_budget_summary)
        deepest_puct_budget = max(puct_budget_summary)
        teacher_prefers_reference = bool(
            teacher_budget_summary[deepest_teacher_budget][
                "teacher_prefers_corrected_reference"
            ]
        )
        neural_prefers_reference = bool(neural_summary["agrees"])
        child_puct_prefers_reference = bool(
            puct_budget_summary[deepest_puct_budget]["puct_prefers_corrected_reference"]
        )
        root_puct_prefers_reference = bool(baseline_1200["pass"])
        reference_suspicious = (
            str(classification["row_classification"])
            == "corrected_reference_suspicious"
        )
        if reference_suspicious:
            bucket = "residual_reference_suspicious"
            recommended_use = "exclude_pending_adjudication"
            suspicious_rows.append(row_id)
        elif (
            teacher_prefers_reference
            and not neural_prefers_reference
            and str(classification["row_classification"]) == "value_head_miscalibration"
        ):
            bucket = "stable_value_head_miscalibration"
            recommended_use = "target_candidate"
        elif (
            teacher_prefers_reference
            and neural_prefers_reference
            and child_puct_prefers_reference
            and not root_puct_prefers_reference
        ):
            bucket = "stable_root_selection_pressure"
            recommended_use = "target_candidate"
        elif teacher_prefers_reference and not child_puct_prefers_reference:
            bucket = "stable_puct_child_mismatch"
            recommended_use = "target_candidate"
        else:
            bucket = "residual_reference_suspicious"
            recommended_use = "holdout"
            suspicious_rows.append(row_id)
        mechanism_rows.append(
            {
                "row_id": row_id,
                "selected_move": baseline_1200["selected_move"],
                "corrected_reference_move": row.corrected_reference_move,
                "row_classification": classification["row_classification"],
                "teacher_child_prefers_reference": teacher_prefers_reference,
                "neural_child_prefers_reference": neural_prefers_reference,
                "child_puct_prefers_reference": child_puct_prefers_reference,
                "root_puct_prefers_reference": root_puct_prefers_reference,
                "reference_suspicious": reference_suspicious,
                "recommended_use": recommended_use,
                "notes": classification["supporting_evidence"],
            }
        )
        bucket_by_row_id[row_id] = bucket
    return {
        "teacher_budgets": teacher_budgets,
        "mechanism_rows": sorted(mechanism_rows, key=lambda row: row["row_id"]),
        "classification_rows": sorted(
            classification_rows, key=lambda row: row["row_id"]
        ),
        "child_neural_rows": child_neural_rows,
        "child_teacher_rows": child_teacher_rows,
        "child_puct_rows": child_puct_rows,
        "counterfactual_table": counterfactual_table,
        "bucket_by_row_id": bucket_by_row_id,
        "suspicious_rows": sorted(set(suspicious_rows)),
    }


def summarize_buckets(
    *,
    mechanism_rows: list[dict[str, Any]],
    bucket_by_row_id: dict[str, str],
    role_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    rows_by_bucket: dict[str, list[str]] = defaultdict(list)
    train_eligible_by_bucket: Counter[str] = Counter()
    holdout_by_bucket: Counter[str] = Counter()
    for row in mechanism_rows:
        row_id = str(row["row_id"])
        bucket = str(bucket_by_row_id.get(row_id, row["row_classification"]))
        rows_by_bucket[bucket].append(row_id)
        if str(row.get("recommended_use")) == "target_candidate":
            train_eligible_by_bucket[bucket] += 1
        if (
            role_by_id.get(row_id) == "holdout_candidate"
            or str(row.get("recommended_use")) == "holdout"
        ):
            holdout_by_bucket[bucket] += 1
    ordered_buckets = [
        "stable_value_head_miscalibration",
        "stable_root_selection_pressure",
        "stable_puct_child_mismatch",
        "residual_reference_suspicious",
        "pass_after_021_update",
        "controls_stable",
    ]
    summary_rows: list[dict[str, Any]] = []
    for bucket in ordered_buckets:
        row_ids = sorted(rows_by_bucket.get(bucket, []))
        if bucket == "stable_value_head_miscalibration":
            risks = "value drift if targets are too small or references regress"
            next_action = "eligible for train-only value calibration artifact"
        elif bucket == "stable_root_selection_pressure":
            risks = "root prior pressure may mask correct child values"
            next_action = "root cpuct/prior diagnostics"
        elif bucket == "stable_puct_child_mismatch":
            risks = (
                "child search mismatch may indicate backup or child-selection issues"
            )
            next_action = "search-stack diagnostics before training"
        elif bucket == "residual_reference_suspicious":
            risks = "incorrect labels would contaminate any training target set"
            next_action = "focused non-mutating adjudication"
        elif bucket == "pass_after_021_update":
            risks = "pass rows should not be treated as failures"
            next_action = "keep as validation/pass rows"
        else:
            risks = "controls must remain stable during any future experiment"
            next_action = "preserve as regression gates"
        summary_rows.append(
            {
                "mechanism": bucket,
                "row_count": len(row_ids),
                "rows": row_ids,
                "train_target_eligible_count": int(train_eligible_by_bucket[bucket]),
                "holdout_candidate_count": int(holdout_by_bucket[bucket]),
                "risks": risks,
                "next_action": next_action,
            }
        )
    return summary_rows


def targetability_decision(
    *,
    mining_summary: dict[str, Any],
    mechanism_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    ranking_rows = [
        row
        for row in list(mining_summary.get("family_rankings") or [])
        if isinstance(row, dict)
    ]
    ranking_rows.sort(key=lambda row: float(row.get("rank_score") or 0.0), reverse=True)
    top_family = str(ranking_rows[0]["family"]) if ranking_rows else FAMILY
    bucket_counts = {
        row["mechanism"]: int(row["row_count"]) for row in mechanism_summary
    }
    stable_value = bucket_counts.get("stable_value_head_miscalibration", 0)
    stable_root = bucket_counts.get("stable_root_selection_pressure", 0)
    stable_puct = bucket_counts.get("stable_puct_child_mismatch", 0)
    residual = bucket_counts.get("residual_reference_suspicious", 0)
    remaining_failures = stable_value + stable_root + stable_puct + residual

    if top_family != FAMILY:
        return {
            "status": "no_longer_top_target_family",
            "recommended_next_action": "rerun corrected non-opening family mining and select the new top family.",
            "notes": f"{FAMILY} no longer ranks first after the corrected 021 rebaseline; top family is {top_family}",
        }
    if residual > max(stable_value, stable_root, stable_puct):
        return {
            "status": "needs_more_reference_adjudication",
            "recommended_next_action": "run focused adjudication on the residual suspicious rows before any training.",
            "notes": "residual suspicious references remain the largest post-adjudication bucket",
        }
    if remaining_failures > 0 and stable_value > remaining_failures / 2.0:
        return {
            "status": "ready_for_value_calibration_artifact",
            "recommended_next_action": "build a small train-only child-afterstate value-calibration artifact for this mechanism bucket, with preservation controls and no arena until local value metrics improve.",
            "notes": "most remaining failures are stable value-head miscalibration rows",
        }
    if remaining_failures > 0 and stable_root > remaining_failures / 2.0:
        return {
            "status": "ready_for_root_selection_calibration",
            "recommended_next_action": "run cpuct/root-prior calibration diagnostics for this bucket, not value training.",
            "notes": "most remaining failures are stable root-selection-pressure rows",
        }
    return {
        "status": "mixed_mechanisms_split_required",
        "recommended_next_action": "split into mechanism-specific target sets and choose the largest stable bucket.",
        "notes": "multiple mechanisms remain after the corrected-reference rebaseline",
    }


def build_report(summary: dict[str, Any]) -> str:
    fixture_rows = summary["fixture_verification_table"]
    baseline_rows = summary["family_baseline_table"]
    compare_row = summary["row_021_comparison_table"][0]
    mechanism_rows = summary["mechanism_table"]
    mechanism_summary_rows = summary["mechanism_summary_table"]
    suspicious_rows = summary["remaining_suspicious_references"]
    decision = summary["targetability_decision"]
    lines = [
        "# AlphaZero-lite Incumbent Proxy Disagreement Post-Adjudication Rebaseline Results",
        "",
        "## 1. Context",
        "",
        "- No training was run.",
        "- No arena was run.",
        "- No model was promoted.",
        "- No replay artifacts were created.",
        f"- Corrected references: `{summary['inputs']['reference_path']}`.",
        f"- Current artifact: `{summary['inputs']['current_artifact']}`.",
        "",
        "## 2. Why PR #43 made the previous audit stale",
        "",
        "- PR #43’s checked-in value/backup audit used the pre-adjudication `incumbent_proxy_disagreement-021` label with reference move `3`.",
        "- The merged approved fixture update changed row `021` to corrected reference move `2`, so the previous family failure inventory and mechanism counts could no longer be trusted as current.",
        "- This rerun recomputes the family baseline and the mechanism audit against the live corrected fixture rather than reusing the stale pre-adjudication report.",
        "",
        "## 3. Fixture verification",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "active_reference_move",
                "expected_reference_move",
                "observed_reference_moves",
                "reference_unstable",
                "canonical_state_match",
                "status",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["active_reference_move"]),
                    str(row["expected_reference_move"]),
                    json.dumps(row["observed_reference_moves"]),
                    format_bool(row["reference_unstable"]),
                    format_bool(row["canonical_state_match"]),
                    row["status"],
                    row["notes"],
                ]
                for row in fixture_rows
            ],
        )
    )
    lines.extend(["", "## 4. Post-adjudication family baseline", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "corrected_reference_move",
                "selected_384",
                "selected_1200",
                "pass_384",
                "pass_1200",
                "reference_visit_share_384",
                "reference_visit_share_1200",
                "selected_minus_reference_q_margin_384",
                "selected_minus_reference_q_margin_1200",
                "severity",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["corrected_reference_move"]),
                    str(row["selected_384"]),
                    str(row["selected_1200"]),
                    format_bool(row["pass_384"]),
                    format_bool(row["pass_1200"]),
                    format_float(row["reference_visit_share_384"]),
                    format_float(row["reference_visit_share_1200"]),
                    format_float(row["selected_minus_reference_q_margin_384"]),
                    format_float(row["selected_minus_reference_q_margin_1200"]),
                    row["severity"],
                    row["notes"],
                ]
                for row in baseline_rows
            ],
        )
    )
    lines.extend(["", "## 5. Row 021 pre-vs-post comparison", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "old_reference_move",
                "new_reference_move",
                "current_selected_384",
                "current_selected_1200",
                "old_label_status",
                "new_label_status",
                "recommended_role",
                "notes",
            ],
            [
                [
                    compare_row["row_id"],
                    str(compare_row["old_reference_move"]),
                    str(compare_row["new_reference_move"]),
                    str(compare_row["current_selected_384"]),
                    str(compare_row["current_selected_1200"]),
                    compare_row["old_label_status"],
                    compare_row["new_label_status"],
                    compare_row["recommended_role"],
                    compare_row["notes"],
                ]
            ],
        )
    )
    lines.extend(["", "## 6. Post-adjudication value/backup audit", ""])
    lines.append(f"- Audited rows: `{summary['audited_row_ids']}`.")
    lines.append(
        f"- Teacher budgets used: `{summary['post_adjudication_audit']['teacher_budgets']}`."
    )
    lines.extend(["", "## 7. Mechanism-specific subfamilies", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "selected_move",
                "corrected_reference_move",
                "row_classification",
                "teacher_child_prefers_reference",
                "neural_child_prefers_reference",
                "child_puct_prefers_reference",
                "root_puct_prefers_reference",
                "reference_suspicious",
                "recommended_use",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["selected_move"]),
                    str(row["corrected_reference_move"]),
                    row["row_classification"],
                    format_bool(row["teacher_child_prefers_reference"]),
                    format_bool(row["neural_child_prefers_reference"]),
                    format_bool(row["child_puct_prefers_reference"]),
                    format_bool(row["root_puct_prefers_reference"]),
                    format_bool(row["reference_suspicious"]),
                    row["recommended_use"],
                    row["notes"],
                ]
                for row in mechanism_rows
            ],
        )
    )
    lines.extend(["", ""])
    lines.extend(
        markdown_table(
            [
                "mechanism",
                "row_count",
                "rows",
                "train_target_eligible_count",
                "holdout_candidate_count",
                "risks",
                "next_action",
            ],
            [
                [
                    row["mechanism"],
                    str(row["row_count"]),
                    ", ".join(row["rows"]),
                    str(row["train_target_eligible_count"]),
                    str(row["holdout_candidate_count"]),
                    row["risks"],
                    row["next_action"],
                ]
                for row in mechanism_summary_rows
            ],
        )
    )
    lines.extend(["", "## 8. Remaining suspicious references", ""])
    if suspicious_rows:
        for row_id in suspicious_rows:
            lines.append(
                f"- `{row_id}` remains excluded from training targets pending focused adjudication."
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## 9. Targetability decision", ""])
    lines.append(f"- Decision: `{decision['status']}`.")
    lines.append(f"- Notes: {decision['notes']}")
    lines.extend(["", "## 10. Exactly one recommended next action", ""])
    lines.append(f"Recommendation: **{decision['recommended_next_action']}**")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    rerun_family_mining(root)

    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_by_id, _reference_by_canonical = load_reference_maps(args.reference_path)
    verification_table, verification_result = fixture_verification(
        suite_by_id=suite_by_id, reference_by_id=reference_by_id
    )
    if not verification_result["ok"]:
        summary = {
            "schema": SCHEMA,
            "inputs": {
                "reference_path": str(args.reference_path),
                "suite_path": str(args.suite_path),
                "current_artifact": str(args.current_artifact),
            },
            "fixture_verification_table": verification_table,
            "status": verification_result["error"],
        }
        write_json(args.summary_out, summary)
        raise SystemExit("reference_fixture_not_updated_correctly")

    mining_summary = load_json(args.mining_summary_path)
    selected_rows = load_jsonl(args.selected_rows_path)
    inventory_rows, selected_by_id, role_by_id, _representative_rows = (
        load_family_context(
            mining_summary=mining_summary,
            selected_rows=selected_rows,
        )
    )
    audit_rows = make_audit_rows(
        inventory_rows=inventory_rows,
        suite_by_id=suite_by_id,
        reference_by_id=reference_by_id,
        role_by_id=role_by_id,
    )
    audit_rows_by_id = {row.row_id: row for row in audit_rows}
    evaluator = ArtifactEvaluator(args.current_artifact)
    baseline_table, baselines, baseline_by_row = family_baseline_table(
        audit_rows=audit_rows,
        evaluator=evaluator,
        seed=int(args.seed),
        cpuct=float(args.cpuct),
    )
    row_021 = audit_rows_by_id["incumbent_proxy_disagreement-021"]
    comparison_021 = row_021_comparison(
        row_021=row_021,
        evaluator=evaluator,
        seed=int(args.seed),
        cpuct=float(args.cpuct),
        selected_by_id=selected_by_id,
    )
    audited_ids = audited_row_ids(
        family_baselines=baseline_table,
        role_by_id=role_by_id,
        old_audit_summary_path=args.old_audit_summary_path,
    )
    audit_summary = run_post_adjudication_audit(
        audit_rows_by_id=audit_rows_by_id,
        baselines=baselines,
        evaluator=evaluator,
        audited_ids=audited_ids,
        seed=int(args.seed),
        cpuct=float(args.cpuct),
    )
    mechanism_summary = summarize_buckets(
        mechanism_rows=audit_summary["mechanism_rows"],
        bucket_by_row_id=audit_summary["bucket_by_row_id"],
        role_by_id=role_by_id,
    )
    decision = targetability_decision(
        mining_summary=mining_summary,
        mechanism_summary=mechanism_summary,
    )
    family_ranking = next(
        (
            row
            for row in list(mining_summary.get("family_rankings") or [])
            if isinstance(row, dict) and str(row.get("family")) == FAMILY
        ),
        {},
    )
    summary = {
        "schema": SCHEMA,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "mining_summary_path": str(args.mining_summary_path),
            "selected_rows_path": str(args.selected_rows_path),
        },
        "guardrails": {
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
            "mutated_corrected_references": False,
        },
        "fixture_verification_table": verification_table,
        "family_context": family_ranking,
        "family_baseline_table": baseline_table,
        "baseline_detail_by_row": baseline_by_row,
        "row_021_comparison_table": [comparison_021],
        "audited_row_ids": audited_ids,
        "post_adjudication_audit": audit_summary,
        "mechanism_table": audit_summary["mechanism_rows"],
        "mechanism_summary_table": mechanism_summary,
        "remaining_suspicious_references": audit_summary["suspicious_rows"],
        "targetability_decision": decision,
    }
    write_json(args.summary_out, summary)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.summary_out),
                "report_path": str(args.report_out),
                "targetability_decision": decision["status"],
                "recommended_next_action": decision["recommended_next_action"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
