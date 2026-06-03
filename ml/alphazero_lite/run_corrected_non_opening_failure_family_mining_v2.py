#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.forensic_suite import load_suite
from ml.alphazero_lite.run_corrected_non_opening_failure_family_mining import (
    DEFAULT_CURRENT_PATH,
    DEFAULT_REBASELINE_ROOT,
    DEFAULT_REFERENCE_ARTIFACT,
    DEFAULT_SUITE_PATH,
    OPENING_SUBFAMILY_DIAGNOSTIC_PATH,
    SEVERITY_SCORE,
    average,
    canonical_reference_metadata,
    current_rows_by_id,
    failure_mode_from_probe,
    family_label_for_row,
    format_float,
    format_ratio,
    inventory_is_fresh,
    load_json,
    load_opening_subfamily_rows,
    probe_position,
    repo_root,
    rerun_rebaseline,
    write_json,
    write_jsonl,
)


DEFAULT_OUT_ROOT = Path("/tmp/azlite_corrected_non_opening_failure_mining_v2")
DEFAULT_SUMMARY_PATH = DEFAULT_OUT_ROOT / "non_opening_failure_family_summary_v2.json"
DEFAULT_SELECTED_ROWS_PATH = (
    DEFAULT_OUT_ROOT / "selected_non_opening_family_rows_v2.jsonl"
)
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-corrected-non-opening-failure-family-mining-v2-results.md"
)
EXCLUDED_OPENING_FAMILIES = {
    "opening_plies_1_8",
    "opening_extra_turn_overbias",
    "opening_edge_move_5_preference",
    "opening_missed_extra_turn_continuation",
}
EXCLUDED_GUARD_ROW_IDS = {
    "capture_available-002",
    "capture_available-003",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
}
REFERENCE_EXCLUSION_REASONS = {
    "reference_unstable",
    "reference_integrity_error",
    "train_only_reference",
    "legacy_reference_source",
    "excluded_diagnostic",
    "do_not_train",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT
    )
    parser.add_argument("--current-path", type=Path, default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--rebaseline-root", type=Path, default=DEFAULT_REBASELINE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--selected-rows-path", type=Path, default=DEFAULT_SELECTED_ROWS_PATH
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def exclusion_reason(row: dict[str, Any], *, family_label: str) -> str | None:
    row_id = str(row["row_id"])
    recommended_use = str(row.get("recommended_use") or "")
    failure_status = str(row.get("failure_status") or "")
    reference_source = str(row.get("reference_source") or "")

    if row.get("corrected_reference_move") is None:
        return "missing_corrected_reference_move"
    if family_label in EXCLUDED_OPENING_FAMILIES:
        return "opening_branch_excluded"
    if row_id in EXCLUDED_GUARD_ROW_IDS:
        return "corrected_guard_context_only"
    if str(row.get("family") or "") == "incumbent_proxy_disagreement":
        return "incumbent_proxy_branch_excluded"
    if bool(row.get("reference_unstable")) or failure_status == "unstable_reference":
        return "reference_unstable"
    if failure_status == "reference_integrity_error":
        return "reference_integrity_error"
    if recommended_use == "train_only":
        return "train_only_reference"
    if reference_source.startswith("legacy") or reference_source == "runtime_fallback":
        return "legacy_reference_source"
    if "excluded_diagnostic" in recommended_use or "excluded_diagnostic" in str(
        row.get("notes") or ""
    ):
        return "excluded_diagnostic"
    if recommended_use.startswith("do_not_train"):
        return "do_not_train"
    return None


def classify_probe(probe: dict[str, Any]) -> str:
    if bool(probe.get("pass")):
        return "pass"
    mode = failure_mode_from_probe(probe)
    if mode == "policy_only":
        return "policy_prior"
    return mode


def dominant_failure_mode(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("failure_mode") or "unknown") for row in rows)
    counts.pop("pass", None)
    counts.pop("unknown", None)
    if not counts:
        return "unknown"
    top_mode, top_count = counts.most_common(1)[0]
    total = sum(counts.values())
    if len(counts) > 1 and top_count < math.ceil(total * 0.6):
        return "mixed"
    return str(top_mode)


def rank_score(summary: dict[str, Any]) -> float:
    dominant_mode = str(summary.get("dominant_failure_mode") or "unknown")
    mechanism_bonus = {
        "value_q": 4.0,
        "policy_prior": 3.0,
        "search_selection": 2.0,
        "mixed": -1.0,
        "unknown": -2.0,
    }.get(dominant_mode, 0.0)
    control_bonus = min(int(summary["pass_rows"]), 4) * 1.5
    return round(
        (int(summary["stable_corrected_reference_count"]) * 6.0)
        + (int(summary["persistent_1200_failures"]) * 5.5)
        + (int(summary["high_severity_count"]) * 3.5)
        + (int(summary["medium_severity_count"]) * 2.0)
        + (control_bonus)
        + (
            abs(float(summary.get("avg_selected_minus_reference_q_margin_1200") or 0.0))
            * 10.0
        )
        + ((1.0 - float(summary.get("avg_reference_visit_share_1200") or 0.0)) * 6.0)
        + mechanism_bonus
        - (int(summary["conflicting_target_count"]) * 8.0)
        - (int(summary["duplicate_canonical_state_count"]) * 1.0)
        - (int(summary["recovered_at_1200"]) * 1.5),
        4,
    )


def classify_family(summary: dict[str, Any]) -> tuple[str, str]:
    if int(summary["stable_corrected_reference_count"]) < int(summary["total_rows"]):
        return (
            "needs_reference_adjudication",
            "family still includes non-stable references",
        )
    if int(summary["conflicting_target_count"]) > 0:
        return "needs_reference_adjudication", "canonical target conflicts detected"
    if int(summary["persistent_1200_failures"]) < 3:
        if int(summary["recovered_at_1200"]) >= max(2, int(summary["fail_rows"]) // 2):
            return "guard_budget_noise", "most failures recover by 1200 simulations"
        return "too_sparse", "fewer than three persistent failures remain at 1200"
    if int(summary["recovered_at_1200"]) >= max(
        3, math.ceil(int(summary["fail_rows"]) * 0.6)
    ):
        return "guard_budget_noise", "most failures recover by 1200 simulations"
    if int(summary["pass_rows"]) < 2:
        return "too_sparse", "not enough same-family passing controls"
    dominant_mode = str(summary["dominant_failure_mode"])
    if dominant_mode == "mixed":
        return (
            "mixed_mechanisms_split_required",
            "persistent failures are mixed across value and selection mechanisms",
        )
    if dominant_mode == "value_q":
        return "value_or_backup_issue", "persistent failures remain Q/value-dominant"
    if dominant_mode == "policy_prior":
        return "policy_prior_issue", "persistent failures remain policy-prior-dominant"
    if dominant_mode == "search_selection":
        return (
            "search_selection_issue",
            "persistent failures remain selection-pressure-dominant",
        )
    return "good_target", "stable repeated failures persist with usable controls"


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -SEVERITY_SCORE.get(str(row.get("severity") or "none"), 0),
        -abs(float(row.get("selected_minus_reference_q_margin_1200") or 0.0)),
        float(row.get("reference_visit_share_1200") or 1.0),
        str(row["row_id"]),
    )


def pass_control_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row.get("reference_visit_share_1200") or 0.0),
        -float(row.get("reference_visit_share_384") or 0.0),
        str(row["row_id"]),
    )


def sample_representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failing = sorted(
        [row for row in rows if row["failure_status"] == "fail_corrected_reference"],
        key=row_sort_key,
    )[:10]
    controls = sorted(
        [row for row in rows if row["failure_status"] == "pass_corrected_reference"],
        key=pass_control_sort_key,
    )[:4]
    if len(controls) < 2:
        return failing + controls
    return failing + controls[: max(2, len(controls))]


def persistent_failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["failure_status"] == "fail_corrected_reference"
        and not bool(row.get("pass_1200"))
    ]


def choose_selected_family(
    family_rankings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    acceptable = {
        "good_target",
        "value_or_backup_issue",
        "policy_prior_issue",
        "search_selection_issue",
        "mixed_mechanisms_split_required",
        "needs_reference_adjudication",
    }
    for family in family_rankings:
        if family["classification"] in acceptable:
            return family
    return None


def selected_rows_for_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    persistent = sorted(persistent_failure_rows(rows), key=row_sort_key)
    controls = sorted(
        [row for row in rows if row["failure_status"] == "pass_corrected_reference"],
        key=pass_control_sort_key,
    )
    holdout_count = 0
    if len(persistent) >= 8:
        holdout_count = 2
    elif len(persistent) >= 5:
        holdout_count = 1
    holdout_ids = (
        {row["row_id"] for row in persistent[-holdout_count:]}
        if holdout_count
        else set()
    )
    selected = []
    for row in persistent:
        selected.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "canonical_state_hash": row["canonical_state_hash"],
                "corrected_reference_move": row["corrected_reference_move"],
                "current_selected_move": row["selected_move_1200"],
                "failure_mode": row["failure_mode_1200"],
                "severity": row["severity"],
                "recommended_role": "holdout_candidate"
                if row["row_id"] in holdout_ids
                else "target_candidate",
                "do_not_train_yet": True,
            }
        )
    for row in controls[:4]:
        selected.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "canonical_state_hash": row["canonical_state_hash"],
                "corrected_reference_move": row["corrected_reference_move"],
                "current_selected_move": row["selected_move_1200"],
                "failure_mode": "pass",
                "severity": row["severity"],
                "recommended_role": "preservation_control",
                "do_not_train_yet": True,
            }
        )
    return selected


def exclusion_summary_rows(excluded_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in excluded_rows:
        grouped[str(row["exclusion_reason"])].append(row)
    labels = [
        (
            "opening_branch_excluded",
            "opening branch",
            "opening replay branch is closed",
            "includes opening_plies_1_8 and named opening subfamilies",
        ),
        (
            "corrected_guard_context_only",
            "corrected guard rows",
            "corrected guard confirmations stay context-only",
            "exclude capture_available 002/003/006/007/008",
        ),
        (
            "incumbent_proxy_branch_excluded",
            "incumbent_proxy branch",
            "incumbent_proxy branch is diagnostic-only/exhausted here",
            "do not continue cross-teacher or soft-policy follow-ups in this task",
        ),
        (
            "reference_unstable",
            "reference instability",
            "reference remains unstable",
            "not safe for training-family selection",
        ),
        (
            "reference_integrity_error",
            "reference integrity",
            "reference integrity error",
            "must rebuild/adjudicate before training use",
        ),
        (
            "missing_corrected_reference_move",
            "missing references",
            "row is missing corrected reference move",
            "exclude rows without usable corrected targets",
        ),
        (
            "train_only_reference",
            "train-only rows",
            "train-only rows excluded",
            "cannot become next selected family",
        ),
        (
            "legacy_reference_source",
            "legacy references",
            "stale or legacy reference source excluded",
            "active corrected references only",
        ),
        (
            "excluded_diagnostic",
            "excluded diagnostic",
            "diagnostic-only rows excluded",
            "kept only for context summary",
        ),
        (
            "do_not_train",
            "do-not-train rows",
            "explicit do-not-train recommendation",
            "kept only for context summary",
        ),
    ]
    rows = []
    for key, label, reason, notes in labels:
        members = grouped.get(key, [])
        if not members:
            continue
        rows.append(
            {
                "exclusion_group": label,
                "row_count": len(members),
                "reason": reason,
                "notes": notes,
            }
        )
    return rows


def representative_row_record(
    row: dict[str, Any], *, include_2400: bool = False
) -> dict[str, Any]:
    notes = []
    if row["failure_status"] == "pass_corrected_reference":
        notes.append("passing_control")
    elif not bool(row.get("pass_1200")):
        notes.append("persists_at_1200")
    else:
        notes.append("384_only_recovers_at_1200")
    if include_2400:
        if not bool(row.get("pass_2400")):
            notes.append("persists_at_2400")
        elif row.get("selected_move_2400") == row.get("corrected_reference_move"):
            notes.append("recovers_at_2400")
    return {
        "family": row["family"],
        "row_id": row["row_id"],
        "corrected_reference_move": row["corrected_reference_move"],
        "selected_move_384": row["selected_move_384"],
        "selected_move_1200": row["selected_move_1200"],
        "selected_move_2400": row.get("selected_move_2400") if include_2400 else None,
        "reference_visit_share_384": row["reference_visit_share_384"],
        "reference_visit_share_1200": row["reference_visit_share_1200"],
        "selected_minus_reference_q_margin_384": row[
            "selected_minus_reference_q_margin_384"
        ],
        "selected_minus_reference_q_margin_1200": row[
            "selected_minus_reference_q_margin_1200"
        ],
        "failure_mode": row["failure_mode_1200"]
        if row["failure_status"] != "pass_corrected_reference"
        else "pass",
        "notes": ",".join(notes),
        "reference_policy_probability": row["reference_policy_probability_1200"],
        "selected_policy_probability": row["selected_policy_probability_1200"],
    }


def decision_for_selection(
    selected_family: dict[str, Any] | None, family_rankings: list[dict[str, Any]]
) -> tuple[str, str]:
    if family_rankings and all(
        row["classification"] in {"too_sparse", "guard_budget_noise"}
        for row in family_rankings[:5]
    ):
        return (
            "guard_budget_noise_dominant",
            "audit local gate budget policy before more training.",
        )
    if selected_family is None:
        return (
            "diffuse_failure_inventory_after_exclusions",
            "improve mining/scoring or revisit teacher-policy architecture, not replay.",
        )
    classification = str(selected_family["classification"])
    family_name = str(selected_family["family"])
    if classification == "value_or_backup_issue":
        return (
            "next_family_value_backup_audit_ready",
            f"run child-afterstate value/backup audit for `{family_name}` before training.",
        )
    if classification == "policy_prior_issue":
        return (
            "next_family_policy_prior_audit_ready",
            f"run root prior/PUCT pressure diagnostics for `{family_name}` before training.",
        )
    if classification in {"good_target", "search_selection_issue"}:
        return (
            "next_family_diagnostic_artifact_ready",
            f"build a tiny train-only diagnostic artifact for `{family_name}` with preservation controls, no arena.",
        )
    if classification == "guard_budget_noise":
        return (
            "guard_budget_noise_dominant",
            "audit local gate budget policy before more training.",
        )
    if classification == "needs_reference_adjudication":
        return (
            "reference_adjudication_needed",
            f"adjudicate `{family_name}` before training.",
        )
    return (
        "diffuse_failure_inventory_after_exclusions",
        "improve mining/scoring or revisit teacher-policy architecture, not replay.",
    )


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Corrected Non-Opening Failure Family Mining v2 Results",
        "",
        "## 1. Context",
        "",
        "- This v2 mining pass selects the next corrected non-opening failure family without training, arena, promotion, or replay-artifact creation.",
        f"- Corrected references default to `{summary['inventory_source']['reference_artifact']}`.",
        f"- Current artifact evaluated: `{summary['inventory_source']['current_artifact']}`.",
        "",
        "## 2. Why Start-State Fairness Did Not Advance",
        "",
        "- PR #50 left default standard Kalah(6,4) self-play unchanged and treated new start modes as diagnostic-only curriculum variants.",
        "- The fairness scan did not find genuinely near-zero symmetric total-24 starts; the best observed absolute first-player margin stayed materially non-zero.",
        "- Tiny traces improved some local disagreement metrics but worsened standard-start calibration, so no production-scale candidate advanced.",
        "- The fairness branch is therefore closed as `start_bias_not_primary_bottleneck` for immediate next-family selection.",
        "",
        "## 3. Why Opening and Incumbent_proxy Branches Are Excluded",
        "",
        "- Opening replay is closed after the prior guard-safe lane failed to produce strength gains; opening rows remain context only.",
        "- Corrected guard confirmation rows remain validation context, not training-family targets.",
        "- `incumbent_proxy_disagreement` remains diagnostic-only after cross-teacher interference and low-LR/soft-policy follow-ups failed the strict non-regression gate.",
        "",
        "## 4. Corrected Failure Inventory Source",
        "",
        f"- Inventory path: `{summary['inventory_source']['inventory_path']}`.",
        f"- Inventory mode: `{summary['inventory_source']['mode']}`.",
        f"- Inventory freshness reason: `{summary['inventory_source']['freshness_reason']}`.",
        f"- Effective corrected reference path: `{summary['inventory_source']['effective_reference_path']}`.",
        f"- Forensic validation path: `{summary['inventory_source']['forensic_validation_path']}`.",
        "",
        "## 5. Exclusion Summary",
        "",
        "| exclusion_group | row_count | reason | notes |",
        "| --- | ---: | --- | --- |",
    ]
    for row in summary["exclusion_summary"]:
        lines.append(
            f"| {row['exclusion_group']} | {row['row_count']} | {row['reason']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 6. Remaining Family Ranking",
            "",
            "| family | rows | failures | failure_rate | persistent_1200_failures | recovered_at_1200 | high_severity | medium_severity | avg_reference_visit_share_1200 | avg_selected_minus_reference_q_margin_1200 | dominant_failure_mode | classification | notes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for family in summary["family_rankings"]:
        lines.append(
            "| {family} | {rows} | {failures} | {failure_rate} | {persistent} | {recovered} | {high_severity} | {medium_severity} | {share_1200} | {gap_1200} | {mode} | {classification} | {notes} |".format(
                family=family["family"],
                rows=family["total_rows"],
                failures=family["fail_rows"],
                failure_rate=format_ratio(family["failure_rate"]),
                persistent=family["persistent_1200_failures"],
                recovered=family["recovered_at_1200"],
                high_severity=family["high_severity_count"],
                medium_severity=family["medium_severity_count"],
                share_1200=format_float(family["avg_reference_visit_share_1200"]),
                gap_1200=format_float(
                    family["avg_selected_minus_reference_q_margin_1200"]
                ),
                mode=family["dominant_failure_mode"],
                classification=family["classification"],
                notes=family["notes"],
            )
        )
    lines.extend(
        [
            "",
            "## 7. Representative Row Diagnostics",
            "",
            "| family | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | selected_move_2400 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | failure_mode | notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in summary["representative_rows"]:
        lines.append(
            "| {family} | {row_id} | {reference_move} | {move_384} | {move_1200} | {move_2400} | {share_384} | {share_1200} | {gap_384} | {gap_1200} | {failure_mode} | {notes} |".format(
                family=row["family"],
                row_id=row["row_id"],
                reference_move=row["corrected_reference_move"],
                move_384=row["selected_move_384"]
                if row["selected_move_384"] is not None
                else "-",
                move_1200=row["selected_move_1200"]
                if row["selected_move_1200"] is not None
                else "-",
                move_2400=row.get("selected_move_2400")
                if row.get("selected_move_2400") is not None
                else "-",
                share_384=format_float(row["reference_visit_share_384"]),
                share_1200=format_float(row["reference_visit_share_1200"]),
                gap_384=format_float(row["selected_minus_reference_q_margin_384"]),
                gap_1200=format_float(row["selected_minus_reference_q_margin_1200"]),
                failure_mode=row["failure_mode"],
                notes=row["notes"],
            )
        )
    lines.extend(
        [
            "",
            "## 8. Targetability Classification",
            "",
        ]
    )
    for family in summary["family_rankings"][:5]:
        lines.append(
            f"- `{family['family']}`: `{family['classification']}`. {family['notes']}"
        )
    lines.extend(
        [
            "",
            "## 9. Selected Next Family",
            "",
            "| selected_family | target_rows | control_rows | holdout_rows | reason_selected | risks | next_action |",
            "| --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    selected = summary["selected_family"]
    lines.append(
        "| {family} | {target_rows} | {control_rows} | {holdout_rows} | {reason} | {risks} | {next_action} |".format(
            family=selected["selected_family"],
            target_rows=selected["target_rows"],
            control_rows=selected["control_rows"],
            holdout_rows=selected["holdout_rows"],
            reason=selected["reason_selected"],
            risks=selected["risks"],
            next_action=selected["next_action"],
        )
    )
    lines.extend(
        [
            "",
            "## 10. Exactly One Recommended Next Action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**",
            "",
            f"Run classification: `{summary['decision_classification']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    inventory_path = args.rebaseline_root / "corrected_failure_inventory.json"
    rebaseline_summary_path = (
        args.rebaseline_root / "forensic_reference_rebaseline_summary.json"
    )
    forensic_validation_path = args.rebaseline_root / "forensic_suite_validation.json"
    effective_reference_path = (
        args.rebaseline_root / "incumbent_forensic_references_v1_rebased.json"
    )
    opening_subfamilies = load_opening_subfamily_rows(OPENING_SUBFAMILY_DIAGNOSTIC_PATH)

    fresh, freshness_reason = inventory_is_fresh(
        inventory_path=inventory_path,
        summary_path=rebaseline_summary_path,
        suite_path=args.suite_path,
        reference_artifact=args.reference_artifact,
    )
    inventory_mode = "loaded_existing"
    if not fresh:
        rerun_rebaseline(
            root=root,
            suite_path=args.suite_path,
            reference_artifact=args.reference_artifact,
            current_path=args.current_path,
            out_root=args.rebaseline_root,
            dry_run=args.dry_run,
        )
        inventory_mode = "rebuilt"
        freshness_reason = f"refreshed_due_to_{freshness_reason}"

    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    inventory_rows = load_json(inventory_path)
    forensic_validation = load_json(forensic_validation_path)
    reference_payload = load_json(effective_reference_path)
    current_by_id = current_rows_by_id(forensic_validation)
    reference_by_id = canonical_reference_metadata(reference_payload)

    evaluator = ArtifactEvaluator(args.current_path)
    probe_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def probe_for(row: dict[str, Any], budget: int) -> dict[str, Any]:
        key = (str(row["row_id"]), int(budget))
        cached = probe_cache.get(key)
        if cached is not None:
            return cached
        cached = probe_position(
            evaluator=evaluator,
            current_path=args.current_path,
            state=dict(row["state"]),
            legal_moves=list(row["legal_moves"]),
            reference_move=int(row["corrected_reference_move"]),
            budget=int(budget),
            seed=int(args.seed),
        )
        probe_cache[key] = cached
        return cached

    enriched_rows: list[dict[str, Any]] = []
    for raw_row in inventory_rows:
        row_id = str(raw_row["row_id"])
        suite_row = suite_by_id[row_id]
        validation_row = current_by_id.get(row_id, {})
        reference_row = reference_by_id.get(row_id, {})
        family = str(raw_row["family"])
        family_label = family_label_for_row(row_id, family, opening_subfamilies)
        enriched = {
            **raw_row,
            "family_label": family_label,
            "state": dict(suite_row.state),
            "legal_moves": list(suite_row.legal_moves),
            "bucket": suite_row.bucket,
            "phase": suite_row.phase,
            "tags": list(suite_row.tags),
            "canonical_state_hash": str(
                validation_row.get("canonical_state")
                or reference_row.get("canonical_state")
                or ""
            ),
            "reference_unstable": bool(
                reference_row.get(
                    "reference_unstable",
                    validation_row.get("reference_unstable", False),
                )
            ),
            "reference_source": reference_row.get("reference_source"),
            "reference_policy_probability_384": None,
            "reference_policy_probability_1200": None,
            "selected_policy_probability_384": None,
            "selected_policy_probability_1200": None,
        }
        enriched["exclusion_reason"] = exclusion_reason(
            enriched, family_label=family_label
        )
        enriched_rows.append(enriched)

    for row in enriched_rows:
        if row.get("corrected_reference_move") is None:
            row["selected_move_384"] = None
            row["selected_move_1200"] = None
            row["pass_384"] = False
            row["pass_1200"] = False
            row["failure_mode_384"] = "unknown"
            row["failure_mode_1200"] = "unknown"
            continue
        probe_384 = probe_for(row, 384)
        probe_1200 = probe_for(row, 1200)
        row["selected_move_384"] = probe_384["selected_move"]
        row["selected_move_1200"] = probe_1200["selected_move"]
        row["reference_visit_share_384"] = probe_384["reference_visit_share"]
        row["reference_visit_share_1200"] = probe_1200["reference_visit_share"]
        row["selected_minus_reference_q_margin_384"] = probe_384[
            "selected_minus_reference_q_margin"
        ]
        row["selected_minus_reference_q_margin_1200"] = probe_1200[
            "selected_minus_reference_q_margin"
        ]
        row["reference_policy_probability_384"] = probe_384[
            "reference_policy_probability"
        ]
        row["reference_policy_probability_1200"] = probe_1200[
            "reference_policy_probability"
        ]
        row["selected_policy_probability_384"] = probe_384[
            "selected_policy_probability"
        ]
        row["selected_policy_probability_1200"] = probe_1200[
            "selected_policy_probability"
        ]
        row["pass_384"] = bool(probe_384["pass"])
        row["pass_1200"] = bool(probe_1200["pass"])
        row["failure_mode_384"] = classify_probe(probe_384)
        row["failure_mode_1200"] = classify_probe(probe_1200)
        row["pass_fail_reason_384"] = probe_384["pass_fail_reason"]
        row["pass_fail_reason_1200"] = probe_1200["pass_fail_reason"]

    excluded_rows = [
        row for row in enriched_rows if row["exclusion_reason"] is not None
    ]
    candidate_rows = [row for row in enriched_rows if row["exclusion_reason"] is None]

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        family_rows[str(row["family"])].append(row)

    family_rankings: list[dict[str, Any]] = []
    for family, rows in sorted(family_rows.items()):
        fail_rows = [
            row for row in rows if row["failure_status"] == "fail_corrected_reference"
        ]
        pass_rows = [
            row for row in rows if row["failure_status"] == "pass_corrected_reference"
        ]
        persistent_failures = [row for row in fail_rows if not bool(row["pass_1200"])]
        recovered_at_1200 = [
            row
            for row in fail_rows
            if not bool(row["pass_384"]) and bool(row["pass_1200"])
        ]
        canonical_to_targets: dict[str, set[int]] = defaultdict(set)
        canonical_counter: Counter[str] = Counter()
        for row in rows:
            canonical = str(row["canonical_state_hash"])
            canonical_counter[canonical] += 1
            canonical_to_targets[canonical].add(int(row["corrected_reference_move"]))
        conflicting_target_count = sum(
            1 for targets in canonical_to_targets.values() if len(targets) > 1
        )
        duplicate_canonical_state_count = sum(
            count - 1 for count in canonical_counter.values() if count > 1
        )
        dominant_mode = dominant_failure_mode(
            [{"failure_mode": row["failure_mode_1200"]} for row in persistent_failures]
        )
        summary_row = {
            "family": family,
            "total_rows": len(rows),
            "fail_rows": len(fail_rows),
            "pass_rows": len(pass_rows),
            "failure_rate": round(len(fail_rows) / float(len(rows)), 4)
            if rows
            else 0.0,
            "high_severity_count": sum(
                1 for row in fail_rows if row["severity"] == "high"
            ),
            "medium_severity_count": sum(
                1 for row in fail_rows if row["severity"] == "medium"
            ),
            "stable_corrected_reference_count": sum(
                1 for row in rows if not bool(row["reference_unstable"])
            ),
            "duplicate_canonical_state_count": duplicate_canonical_state_count,
            "conflicting_target_count": conflicting_target_count,
            "avg_reference_visit_share_384": average(
                [row.get("reference_visit_share_384") for row in rows]
            ),
            "avg_reference_visit_share_1200": average(
                [row.get("reference_visit_share_1200") for row in rows]
            ),
            "avg_selected_minus_reference_q_margin_384": average(
                [row.get("selected_minus_reference_q_margin_384") for row in fail_rows]
            ),
            "avg_selected_minus_reference_q_margin_1200": average(
                [
                    row.get("selected_minus_reference_q_margin_1200")
                    for row in persistent_failures
                ]
            ),
            "persistent_1200_failures": len(persistent_failures),
            "recovered_at_1200": len(recovered_at_1200),
            "dominant_failure_mode": dominant_mode,
        }
        classification, notes = classify_family(summary_row)
        summary_row["classification"] = classification
        summary_row["notes"] = notes
        summary_row["rank_score"] = rank_score(summary_row)
        family_rankings.append(summary_row)

    family_rankings.sort(
        key=lambda row: (
            -float(row["rank_score"]),
            -int(row["persistent_1200_failures"]),
            -int(row["fail_rows"]),
            str(row["family"]),
        )
    )

    representative_rows: list[dict[str, Any]] = []
    top_families = family_rankings[:5]
    leading_family = top_families[0]["family"] if top_families else None
    for family_summary in top_families:
        family_name = str(family_summary["family"])
        rows = family_rows[family_name]
        for row in sample_representatives(rows):
            include_2400 = family_name == leading_family
            if include_2400:
                probe_2400 = probe_for(row, 2400)
                row["selected_move_2400"] = probe_2400["selected_move"]
                row["pass_2400"] = bool(probe_2400["pass"])
            representative_rows.append(
                representative_row_record(row, include_2400=include_2400)
            )

    selected_family_summary = choose_selected_family(family_rankings)
    selected_family_name = (
        str(selected_family_summary["family"])
        if selected_family_summary
        else "none_safe"
    )
    decision_classification, recommended_next_action = decision_for_selection(
        selected_family_summary, family_rankings
    )

    selected_family_rows: list[dict[str, Any]] = []
    selected_family_report = {
        "selected_family": "none_safe",
        "target_rows": 0,
        "control_rows": 0,
        "holdout_rows": 0,
        "reason_selected": "no coherent non-opening family remained safe after exclusions",
        "risks": "diffuse, sparse, or reference-sensitive failures",
        "next_action": recommended_next_action,
    }
    if selected_family_summary is not None:
        selected_family_rows = selected_rows_for_family(
            family_rows[str(selected_family_summary["family"])]
        )
        selected_family_report = {
            "selected_family": selected_family_name,
            "target_rows": sum(
                1
                for row in selected_family_rows
                if row["recommended_role"] == "target_candidate"
            ),
            "control_rows": sum(
                1
                for row in selected_family_rows
                if row["recommended_role"] == "preservation_control"
            ),
            "holdout_rows": sum(
                1
                for row in selected_family_rows
                if row["recommended_role"] == "holdout_candidate"
            ),
            "reason_selected": (
                "highest-ranked remaining corrected non-opening family after exclusions; "
                + str(selected_family_summary["notes"])
            ),
            "risks": (
                f"dominant mechanism: {selected_family_summary['dominant_failure_mode']}"
            ),
            "next_action": recommended_next_action,
        }

    summary = {
        "schema": "azlite_corrected_non_opening_failure_family_mining_v2",
        "inventory_source": {
            "inventory_path": str(inventory_path),
            "forensic_validation_path": str(forensic_validation_path),
            "effective_reference_path": str(effective_reference_path),
            "reference_artifact": str(args.reference_artifact),
            "current_artifact": str(args.current_path),
            "mode": inventory_mode,
            "freshness_reason": freshness_reason,
        },
        "inventory_rows": [
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "family_label": row["family_label"],
                "corrected_reference_move": row.get("corrected_reference_move"),
                "current_selected_move_384": row.get("selected_move_384"),
                "current_selected_move_1200": row.get("selected_move_1200"),
                "failure_status": row.get("failure_status"),
                "severity": row.get("severity"),
                "reference_unstable": row.get("reference_unstable"),
                "recommended_use": row.get("recommended_use"),
                "selected_minus_reference_q_margin_384": row.get(
                    "selected_minus_reference_q_margin_384"
                ),
                "selected_minus_reference_q_margin_1200": row.get(
                    "selected_minus_reference_q_margin_1200"
                ),
                "reference_visit_share_384": row.get("reference_visit_share_384"),
                "reference_visit_share_1200": row.get("reference_visit_share_1200"),
                "failure_mode_384": row.get("failure_mode_384"),
                "failure_mode_1200": row.get("failure_mode_1200"),
                "reference_source": row.get("reference_source"),
                "canonical_state_hash": row.get("canonical_state_hash"),
                "exclusion_reason": row.get("exclusion_reason"),
            }
            for row in enriched_rows
        ],
        "exclusion_summary": exclusion_summary_rows(excluded_rows),
        "family_rankings": family_rankings,
        "representative_rows": representative_rows,
        "selected_family": selected_family_report,
        "decision_classification": decision_classification,
        "recommended_next_action": recommended_next_action,
    }

    write_json(args.summary_path, summary)
    write_jsonl(args.selected_rows_path, selected_family_rows)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.summary_path),
                "selected_rows_path": str(args.selected_rows_path),
                "report_path": str(args.report_path),
                "decision_classification": decision_classification,
                "selected_family": selected_family_name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
