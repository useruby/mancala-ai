#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.forensic_suite import load_suite


DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_CURRENT_PATH = Path("storage/ai/alphazero_lite/current")
DEFAULT_REBASELINE_ROOT = Path("/tmp/azlite_forensic_reference_rebaseline")
DEFAULT_OUT_ROOT = Path("/tmp/azlite_corrected_non_opening_failure_mining")
DEFAULT_SUMMARY_PATH = DEFAULT_OUT_ROOT / "non_opening_failure_family_summary.json"
DEFAULT_SELECTED_ROWS_PATH = DEFAULT_OUT_ROOT / "selected_non_opening_family_rows.jsonl"
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-corrected-non-opening-failure-family-mining-results.md"
)
OPENING_SUBFAMILY_DIAGNOSTIC_PATH = (
    DEFAULT_REBASELINE_ROOT / "opening_plies_subfamily_diagnostic.json"
)
SEARCH_OPTIONS = {
    "fpu_mode": "parent_q",
    "reuse_subtree": True,
    "normalize_values": True,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.1,
}
EXCLUDED_OPENING_SUBFAMILIES = {
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
SEVERITY_SCORE = {"high": 2, "medium": 1, "low": 0, "none": 0}


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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def average(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / float(len(numeric)), 4)


def max_mtime(paths: list[Path]) -> float:
    return max(path.stat().st_mtime for path in paths if path.exists())


def inventory_is_fresh(
    *,
    inventory_path: Path,
    summary_path: Path,
    suite_path: Path,
    reference_artifact: Path,
) -> tuple[bool, str]:
    if not inventory_path.exists():
        return False, "inventory_missing"
    if not summary_path.exists():
        return False, "summary_missing"
    try:
        summary = load_json(summary_path)
        inventory = load_json(inventory_path)
    except (OSError, json.JSONDecodeError):
        return False, "inventory_unreadable"
    if not isinstance(inventory, list):
        return False, "inventory_not_list"
    if summary.get("effective_reference_artifact") != str(
        DEFAULT_REBASELINE_ROOT / "incumbent_forensic_references_v1_rebased.json"
    ):
        return False, "unexpected_effective_reference_path"
    if summary.get("reference_integrity_error_count") is None:
        return False, "summary_missing_counts"
    expected_positions = len(load_suite(suite_path))
    if len(inventory) != expected_positions:
        return False, "inventory_row_count_mismatch"
    newest_input = max_mtime([suite_path, reference_artifact])
    if inventory_path.stat().st_mtime < newest_input:
        return False, "inventory_older_than_inputs"
    return True, "fresh"


def rerun_rebaseline(
    *,
    root: Path,
    suite_path: Path,
    reference_artifact: Path,
    current_path: Path,
    out_root: Path,
    dry_run: bool,
) -> None:
    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.run_forensic_reference_rebaseline",
        "--suite-path",
        str(suite_path),
        "--reference-artifact",
        str(reference_artifact),
        "--current-path",
        str(current_path),
        "--out-root",
        str(out_root),
    ]
    if dry_run:
        raise SystemExit("rebaseline refresh required, but --dry-run was set")
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def load_opening_subfamily_rows(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}
    result = {}
    for row in rows:
        if isinstance(row, dict) and row.get("row_id") and row.get("subfamily"):
            result[str(row["row_id"])] = str(row["subfamily"])
    return result


def canonical_reference_metadata(
    reference_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = reference_payload.get("rows") or []
    return {
        str(row["id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("id") is not None
    }


def current_rows_by_id(validation_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    current = ((validation_payload.get("systems") or {}).get("current") or {}).get(
        "rows"
    ) or []
    return {
        str(row["id"]): row
        for row in current
        if isinstance(row, dict) and row.get("id") is not None
    }


def failure_mode_from_probe(probe: dict[str, Any]) -> str:
    pass_fail_reason = str(probe.get("pass_fail_reason") or "")
    if pass_fail_reason in {
        "q_favors_selected",
        "selected_move_beats_reference_on_q_and_u",
    }:
        return "value_q"
    if pass_fail_reason == "u_favors_selected":
        return "search_selection"
    if pass_fail_reason == "policy_prior_favors_selected":
        return "policy_only"
    return "unknown"


def dominant_failure_mode(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("failure_mode") or "unknown") for row in rows)
    counts.pop("pass", None)
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def family_label_for_row(
    row_id: str, family: str, opening_subfamilies: dict[str, str]
) -> str:
    opening_subfamily = opening_subfamilies.get(row_id)
    if opening_subfamily in EXCLUDED_OPENING_SUBFAMILIES:
        return opening_subfamily
    return family


def exclusion_reason(row: dict[str, Any], *, family_label: str) -> str | None:
    row_id = str(row["row_id"])
    if family_label == "opening_plies_1_8":
        return "closed_opening_branch"
    if family_label in EXCLUDED_OPENING_SUBFAMILIES:
        return "closed_opening_subfamily"
    if row_id in EXCLUDED_GUARD_ROW_IDS:
        return "guard_context_only"
    if bool(row.get("reference_unstable")):
        return "reference_unstable"
    if row.get("recommended_use") == "train_only":
        return "train_only_reference"
    if row.get("failure_status") == "reference_integrity_error":
        return "reference_integrity_error"
    if str(row.get("reference_source") or "").startswith("legacy"):
        return "train_only_reference"
    return None


def policy_probability_by_move(
    summary: dict[str, Any], legal_moves: list[int]
) -> dict[int, float]:
    raw_policy = list(summary.get("policy") or [])
    return {
        int(move): round(float(raw_policy[int(move)]), 4)
        for move in legal_moves
        if int(move) < len(raw_policy)
    }


def selection_entry_map(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    selection_breakdown = summary.get("selection_breakdown") or {}
    return {
        int(entry["move"]): entry
        for entry in list(selection_breakdown.get("moves") or [])
        if isinstance(entry, dict) and entry.get("move") is not None
    }


def visit_share(visits: list[float], move: int) -> float | None:
    total = sum(float(value) for value in visits)
    if total <= 0 or move >= len(visits):
        return None
    return round(float(visits[move]) / float(total), 4)


def probe_position(
    *,
    evaluator: ArtifactEvaluator,
    current_path: Path,
    state: dict[str, Any],
    legal_moves: list[int],
    reference_move: int,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    summary = evaluate_artifact_position(
        artifact_path=current_path,
        evaluator=evaluator,
        state=dict(state),
        simulations=int(budget),
        seed=int(seed),
        c_puct=1.25,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )
    selected_move = summary.get("selected_move")
    visits = [float(value) for value in list(summary.get("visits") or [])]
    child_stats = {
        int(child["move"]): child for child in list(summary.get("child_stats") or [])
    }
    policy = policy_probability_by_move(summary, legal_moves)
    reference_q = None
    if reference_move in child_stats:
        reference_q = round(float(child_stats[reference_move].get("q_value", 0.0)), 4)
    selected_q = None
    if selected_move is not None and int(selected_move) in child_stats:
        selected_q = round(
            float(child_stats[int(selected_move)].get("q_value", 0.0)), 4
        )
    selection_entries = selection_entry_map(summary)
    reference_u = None
    if reference_move in selection_entries:
        raw_reference_u = selection_entries[reference_move].get("u_component")
        if raw_reference_u is not None:
            reference_u = round(float(raw_reference_u), 4)
    selected_u = None
    if selected_move is not None and int(selected_move) in selection_entries:
        raw_selected_u = selection_entries[int(selected_move)].get("u_component")
        if raw_selected_u is not None:
            selected_u = round(float(raw_selected_u), 4)
    q_margin = None
    if reference_q is not None and selected_q is not None:
        q_margin = round(float(selected_q) - float(reference_q), 4)
    pass_fail_reason = "selected_move_not_reference"
    if selected_move == reference_move:
        pass_fail_reason = "pass_reference_selected"
    elif (
        reference_q is not None and selected_q is not None and selected_q > reference_q
    ):
        if (
            reference_u is not None
            and selected_u is not None
            and selected_u > reference_u
        ):
            pass_fail_reason = "selected_move_beats_reference_on_q_and_u"
        else:
            pass_fail_reason = "q_favors_selected"
    elif (
        reference_u is not None and selected_u is not None and selected_u > reference_u
    ):
        pass_fail_reason = "u_favors_selected"
    elif (
        selected_move is not None
        and policy.get(int(selected_move)) is not None
        and policy.get(reference_move) is not None
        and float(policy[int(selected_move)]) > float(policy[reference_move])
    ):
        pass_fail_reason = "policy_prior_favors_selected"
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "reference_visit_share": visit_share(visits, reference_move),
        "selected_minus_reference_q_margin": q_margin,
        "reference_policy_probability": policy.get(reference_move),
        "selected_policy_probability": None
        if selected_move is None
        else policy.get(int(selected_move)),
        "q_reference": reference_q,
        "q_selected": selected_q,
        "u_reference": reference_u,
        "u_selected": selected_u,
        "pass": bool(selected_move == reference_move),
        "pass_fail_reason": pass_fail_reason,
    }


def rank_score(summary: dict[str, Any]) -> float:
    avg_gap = abs(float(summary.get("avg_selected_minus_reference_q_margin") or 0.0))
    avg_visit_share = float(summary.get("avg_reference_visit_share") or 0.0)
    coverage_bonus = min(int(summary["total_rows"]), 12)
    return round(
        (int(summary["stable_reference_rows"]) * 5.0)
        + (int(summary["fail_rows"]) * 4.0)
        + (int(summary["high_severity_count"]) * 3.0)
        + (int(summary["medium_severity_count"]) * 1.5)
        + coverage_bonus
        + (avg_gap * 10.0)
        + ((1.0 - avg_visit_share) * 5.0)
        - (int(summary["conflicting_target_count"]) * 6.0)
        - (int(summary["duplicate_canonical_state_count"]) * 1.0),
        4,
    )


def classify_family_targetability(
    *,
    family_summary: dict[str, Any],
    representative_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    fail_rows = int(family_summary["fail_rows"])
    stable_rows = int(family_summary["stable_reference_rows"])
    conflicts = int(family_summary["conflicting_target_count"])
    if stable_rows < family_summary["total_rows"]:
        return (
            "needs_reference_adjudication",
            "family still includes unstable references",
        )
    if conflicts > 0:
        return "needs_reference_adjudication", "canonical target conflicts detected"
    if fail_rows < 3:
        return "too_sparse", "fewer than three corrected failures remain"
    failing_samples = [
        row
        for row in representative_rows
        if row["inventory_failure_status"].startswith("fail")
    ]
    persistent_failures = [
        row
        for row in failing_samples
        if not bool(row.get("pass_384")) and not bool(row.get("pass_1200"))
    ]
    recovered_at_1200 = [
        row
        for row in failing_samples
        if not bool(row.get("pass_384")) and bool(row.get("pass_1200"))
    ]
    mode_counts = Counter(
        str(row.get("failure_mode") or "unknown") for row in persistent_failures
    )
    persistent_count = len(persistent_failures)
    if failing_samples and len(recovered_at_1200) >= math.ceil(
        len(failing_samples) * 0.7
    ):
        return "search_only_issue", "most sampled failures recover at 1200 sims"
    if persistent_count and mode_counts.get("value_q", 0) >= max(
        mode_counts.get("policy_only", 0), mode_counts.get("search_selection", 0), 2
    ):
        return "value_or_backup_issue", "persistent failures remain Q/value-dominant"
    if persistent_count and mode_counts.get("policy_only", 0) >= max(
        mode_counts.get("value_q", 0), mode_counts.get("search_selection", 0), 2
    ):
        return "policy_prior_issue", "persistent failures remain policy-prior-dominant"
    if fail_rows >= 4 and persistent_count >= max(2, min(4, fail_rows // 2)):
        return (
            "good_target",
            "stable repeated failures persist at 1200 with enough rows",
        )
    return "too_sparse", "not enough persistent repeated failures after sampling"


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -SEVERITY_SCORE.get(str(row.get("severity") or "none"), 0),
        -abs(float(row.get("selected_minus_reference_q_margin_384") or 0.0)),
        float(row.get("reference_visit_share_384") or 1.0),
        -float(row.get("regret") or 0.0),
        str(row["row_id"]),
    )


def pass_control_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row.get("reference_visit_share_384") or 0.0) * -1.0,
        float(row.get("reference_visit_share_1200") or 0.0) * -1.0,
        str(row["row_id"]),
    )


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Corrected Non-Opening Failure Family Mining Results",
        "",
        "## 1. Context",
        "",
        "- This run mines corrected non-opening failure families against `storage/ai/alphazero_lite/current`.",
        f"- Corrected references default to `{summary['inventory_source']['reference_artifact']}`.",
        f"- Current artifact evaluated: `{summary['inventory_source']['current_artifact']}`.",
        "",
        "## 2. Why Opening Replay Is Closed",
        "",
        "- PR #41 closed the denoised opening replay lane as `guard_safe_no_strength_gain`.",
        "- Guard-safe candidates lost all 60 arena games against the current artifact.",
        "- Opening replay rows are therefore context only and are excluded from next-family selection.",
        "",
        "## 3. Corrected Failure Inventory Source",
        "",
        f"- Inventory path: `{summary['inventory_source']['inventory_path']}`.",
        f"- Inventory mode: `{summary['inventory_source']['mode']}`.",
        f"- Inventory freshness reason: `{summary['inventory_source']['freshness_reason']}`.",
        f"- Forensic validation path: `{summary['inventory_source']['forensic_validation_path']}`.",
        f"- Opening subfamily diagnostic path: `{summary['inventory_source']['opening_subfamily_diagnostic_path']}`.",
        "",
        "## 4. Exclusions",
        "",
        f"- Excluded rows kept only as context: `{summary['excluded_context']['excluded_row_count']}`.",
        f"- Closed opening rows: `{summary['excluded_context']['closed_opening_row_count']}`.",
        f"- Guard blocker rows excluded from targeting: `{summary['excluded_context']['guard_context_row_count']}`.",
        f"- Reference instability or integrity exclusions: `{summary['excluded_context']['reference_exclusion_row_count']}`.",
        "",
        "## 5. Non-Opening Family Ranking",
        "",
        "| family | rows | failures | failure_rate | high_severity | medium_severity | stable_reference_rows | avg_reference_visit_share | avg_selected_minus_reference_q_margin | classification | notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for family in summary["family_rankings"]:
        lines.append(
            "| {family} | {rows} | {failures} | {failure_rate} | {high_severity} | {medium_severity} | {stable_reference_rows} | {avg_reference_visit_share} | {avg_gap} | {classification} | {notes} |".format(
                family=family["family"],
                rows=family["total_rows"],
                failures=family["fail_rows"],
                failure_rate=format_ratio(family["failure_rate"]),
                high_severity=family["high_severity_count"],
                medium_severity=family["medium_severity_count"],
                stable_reference_rows=family["stable_reference_rows"],
                avg_reference_visit_share=format_float(
                    family["avg_reference_visit_share"]
                ),
                avg_gap=format_float(family["avg_selected_minus_reference_q_margin"]),
                classification=family["classification"],
                notes=family["notes"],
            )
        )
    lines.extend(
        [
            "",
            "## 6. Representative Row Diagnostics",
            "",
            "| family | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | failure_mode | notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in summary["representative_rows"]:
        lines.append(
            "| {family} | {row_id} | {reference_move} | {move_384} | {move_1200} | {share_384} | {share_1200} | {gap_384} | {gap_1200} | {failure_mode} | {notes} |".format(
                family=row["family"],
                row_id=row["row_id"],
                reference_move=row["corrected_reference_move"],
                move_384=row["selected_move_384"]
                if row["selected_move_384"] is not None
                else "-",
                move_1200=row["selected_move_1200"]
                if row["selected_move_1200"] is not None
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
            "## 7. Targetability Classification",
            "",
        ]
    )
    for family in summary["family_rankings"][:3]:
        lines.append(
            f"- `{family['family']}`: `{family['classification']}`. {family['notes']}"
        )
    lines.extend(
        [
            "",
            "## 8. Selected Next Candidate Family",
            "",
            "| selected_family | target_rows | control_rows | holdout_rows | reason_selected | risks | next_action |",
            "| --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    selected = summary["selected_family"]
    lines.append(
        "| {family} | {target_rows} | {control_rows} | {holdout_rows} | {reason_selected} | {risks} | {next_action} |".format(
            family=selected["selected_family"],
            target_rows=selected["target_rows"],
            control_rows=selected["control_rows"],
            holdout_rows=selected["holdout_rows"],
            reason_selected=selected["reason_selected"],
            risks=selected["risks"],
            next_action=selected["next_action"],
        )
    )
    lines.extend(
        [
            "",
            "## 9. Exactly One Recommended Next Action",
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

    enriched_rows: list[dict[str, Any]] = []
    for raw_row in inventory_rows:
        row_id = str(raw_row["row_id"])
        suite_row = suite_by_id[row_id]
        validation_row = current_by_id.get(row_id, {})
        reference_row = reference_by_id.get(row_id, {})
        family = str(raw_row["family"])
        family_label = family_label_for_row(row_id, family, opening_subfamilies)
        source_report_path = str(forensic_validation_path)
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
            "source_report_path": source_report_path,
            "regret": validation_row.get("regret"),
            "value_error": validation_row.get("value_error"),
            "teacher_value": validation_row.get("teacher_value"),
            "system_value": validation_row.get("system_value"),
        }
        enriched["exclusion_reason"] = exclusion_reason(
            enriched, family_label=family_label
        )
        enriched_rows.append(enriched)

    excluded_rows = [
        row for row in enriched_rows if row["exclusion_reason"] is not None
    ]
    candidate_rows = [row for row in enriched_rows if row["exclusion_reason"] is None]

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

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        probe_384 = probe_for(row, 384)
        enriched_probe_row = {
            **row,
            "selected_move_384": probe_384["selected_move"],
            "reference_visit_share_384": probe_384["reference_visit_share"],
            "selected_minus_reference_q_margin_384": probe_384[
                "selected_minus_reference_q_margin"
            ],
            "pass_384": probe_384["pass"],
            "failure_mode_384": failure_mode_from_probe(probe_384),
            "pass_fail_reason_384": probe_384["pass_fail_reason"],
        }
        family_rows[str(row["family"])].append(enriched_probe_row)

    family_rankings: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    sampled_by_family: dict[str, list[dict[str, Any]]] = {}
    for family, rows in sorted(family_rows.items()):
        fail_subset = [
            row
            for row in rows
            if str(row["failure_status"]) == "fail_corrected_reference"
        ]
        pass_subset = [
            row
            for row in rows
            if str(row["failure_status"]) == "pass_corrected_reference"
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
        summary_row = {
            "family": family,
            "total_rows": len(rows),
            "fail_rows": len(fail_subset),
            "pass_rows": len(pass_subset),
            "failure_rate": round(len(fail_subset) / float(len(rows)), 4)
            if rows
            else 0.0,
            "high_severity_count": sum(
                1 for row in fail_subset if row["severity"] == "high"
            ),
            "medium_severity_count": sum(
                1 for row in fail_subset if row["severity"] == "medium"
            ),
            "stable_reference_rows": sum(
                1 for row in rows if not bool(row["reference_unstable"])
            ),
            "duplicate_canonical_state_count": duplicate_canonical_state_count,
            "conflicting_target_count": conflicting_target_count,
            "avg_reference_visit_share": average(
                [row.get("reference_visit_share_384") for row in rows]
            ),
            "avg_selected_minus_reference_q_margin": average(
                [
                    row.get("selected_minus_reference_q_margin_384")
                    for row in fail_subset
                ]
            ),
            "failure_mechanism": dominant_failure_mode(
                [
                    {**row, "failure_mode": row.get("failure_mode_384")}
                    for row in fail_subset
                ]
            ),
        }
        summary_row["rank_score"] = rank_score(summary_row)
        family_rankings.append(summary_row)

    family_rankings.sort(
        key=lambda row: (
            -float(row["rank_score"]),
            -int(row["fail_rows"]),
            str(row["family"]),
        )
    )

    for family_summary in family_rankings[:3]:
        family = str(family_summary["family"])
        rows = family_rows[family]
        fail_subset = sorted(
            [
                row
                for row in rows
                if row["failure_status"] == "fail_corrected_reference"
            ],
            key=row_sort_key,
        )[:10]
        pass_subset = sorted(
            [
                row
                for row in rows
                if row["failure_status"] == "pass_corrected_reference"
            ],
            key=pass_control_sort_key,
        )[:2]
        sampled = fail_subset + pass_subset
        sampled_by_family[family] = []
        for index, row in enumerate(sampled):
            probe_1200 = probe_for(row, 1200)
            sample_row = {
                "family": family,
                "row_id": row["row_id"],
                "canonical_state_hash": row["canonical_state_hash"],
                "corrected_reference_move": row["corrected_reference_move"],
                "selected_move_384": row["selected_move_384"],
                "selected_move_1200": probe_1200["selected_move"],
                "reference_visit_share_384": row["reference_visit_share_384"],
                "reference_visit_share_1200": probe_1200["reference_visit_share"],
                "selected_minus_reference_q_margin_384": row[
                    "selected_minus_reference_q_margin_384"
                ],
                "selected_minus_reference_q_margin_1200": probe_1200[
                    "selected_minus_reference_q_margin"
                ],
                "pass_384": bool(row["pass_384"]),
                "pass_1200": bool(probe_1200["pass"]),
                "failure_mode": "pass"
                if row["failure_status"] == "pass_corrected_reference"
                else failure_mode_from_probe(probe_1200)
                if not probe_1200["pass"]
                else "search_selection",
                "inventory_failure_status": row["failure_status"],
                "severity": row["severity"],
                "corrected_reference_metadata": {
                    "reference_source": row.get("reference_source"),
                    "teacher_value": row.get("teacher_value"),
                    "source_report_path": row.get("source_report_path"),
                },
                "notes": "passing_control"
                if row["failure_status"] == "pass_corrected_reference"
                else "persists_at_1200"
                if not probe_1200["pass"]
                else "384_only_recovers_at_1200",
                "sample_order": index,
            }
            sampled_by_family[family].append(sample_row)
            representative_rows.append(sample_row)

    family_classifications: dict[str, tuple[str, str]] = {}
    for family_summary in family_rankings:
        family = str(family_summary["family"])
        reps = sampled_by_family.get(family, [])
        classification, notes = classify_family_targetability(
            family_summary=family_summary,
            representative_rows=reps,
        )
        family_summary["classification"] = classification
        family_summary["notes"] = notes
        family_classifications[family] = (classification, notes)

    top_family_summary = family_rankings[0] if family_rankings else None
    selected_family_summary = next(
        (row for row in family_rankings if row["classification"] == "good_target"), None
    )
    decision_classification = "diffuse_failure_inventory"
    recommended_next_action = (
        "improve non-opening family mining/scoring before training."
    )
    selected_family_name = "none"
    selected_family_rows: list[dict[str, Any]] = []
    selected_family_report = {
        "selected_family": "none",
        "target_rows": 0,
        "control_rows": 0,
        "holdout_rows": 0,
        "reason_selected": "no family cleared the targetability bar",
        "risks": "diffuse or weakly persistent failures",
        "next_action": "improve mining/scoring rather than training",
    }

    top_families = family_rankings[:3]
    if top_family_summary is not None and selected_family_summary is None:
        selected_family_summary = top_family_summary
    if selected_family_summary is not None:
        selected_family_name = str(selected_family_summary["family"])
    if (
        top_families
        and selected_family_summary is not None
        and selected_family_summary.get("classification") == "good_target"
    ):
        decision_classification = "corrected_non_opening_family_selected"
        recommended_next_action = f"build a small train-only targeted artifact for `{selected_family_name}` with same-family preservation controls and a pre-arena family-specific kill gate."
    elif top_families and any(
        row["classification"] == "needs_reference_adjudication" for row in top_families
    ):
        decision_classification = "reference_adjudication_needed"
        recommended_next_action = (
            f"adjudicate `{selected_family_name}` before training."
        )
    elif top_families and top_families[0]["classification"] == "value_or_backup_issue":
        decision_classification = "value_backup_gap"
        recommended_next_action = (
            f"run a child-afterstate value/backup audit on `{selected_family_name}`."
        )
    elif top_families and top_families[0]["classification"] == "policy_prior_issue":
        decision_classification = "policy_prior_gap"
        recommended_next_action = f"build a small policy-target artifact for `{selected_family_name}` with same-family preservation controls."
    elif top_families and all(
        row["classification"] == "search_only_issue" for row in top_families
    ):
        decision_classification = "guard_budget_noise_dominant"
        recommended_next_action = (
            "audit whether 384-sim local gates are too brittle before more training."
        )

    if selected_family_summary is not None:
        selected_rows = family_rows[selected_family_name]
        failing_rows = sorted(
            [
                row
                for row in selected_rows
                if row["failure_status"] == "fail_corrected_reference"
            ],
            key=row_sort_key,
        )
        passing_rows = sorted(
            [
                row
                for row in selected_rows
                if row["failure_status"] == "pass_corrected_reference"
            ],
            key=pass_control_sort_key,
        )
        holdout_count = 0
        if len(failing_rows) >= 8:
            holdout_count = 2
        elif len(failing_rows) >= 5:
            holdout_count = 1
        holdout_ids = (
            {row["row_id"] for row in failing_rows[-holdout_count:]}
            if holdout_count > 0
            else set()
        )
        control_rows = passing_rows[: max(2, min(4, len(passing_rows)))]
        for row in failing_rows:
            role = (
                "holdout_candidate"
                if row["row_id"] in holdout_ids
                else "target_candidate"
            )
            selected_family_rows.append(
                {
                    "row_id": row["row_id"],
                    "family": selected_family_name,
                    "canonical_state_hash": row["canonical_state_hash"],
                    "corrected_reference_move": row["corrected_reference_move"],
                    "current_selected_move": row["selected_move_384"],
                    "failure_mode": row["failure_mode_384"],
                    "severity": row["severity"],
                    "recommended_role": role,
                }
            )
        for row in control_rows:
            selected_family_rows.append(
                {
                    "row_id": row["row_id"],
                    "family": selected_family_name,
                    "canonical_state_hash": row["canonical_state_hash"],
                    "corrected_reference_move": row["corrected_reference_move"],
                    "current_selected_move": row["selected_move_384"],
                    "failure_mode": "pass",
                    "severity": row["severity"],
                    "recommended_role": "preservation_control",
                }
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
                f"highest-ranked non-opening family; {selected_family_summary['notes']}"
            ),
            "risks": f"dominant mechanism: {selected_family_summary['failure_mechanism']}",
            "next_action": recommended_next_action,
        }

    summary = {
        "schema": "azlite_corrected_non_opening_failure_family_mining_v1",
        "inventory_source": {
            "inventory_path": str(inventory_path),
            "forensic_validation_path": str(forensic_validation_path),
            "effective_reference_path": str(effective_reference_path),
            "reference_artifact": str(args.reference_artifact),
            "current_artifact": str(args.current_path),
            "opening_subfamily_diagnostic_path": str(OPENING_SUBFAMILY_DIAGNOSTIC_PATH),
            "mode": inventory_mode,
            "freshness_reason": freshness_reason,
        },
        "excluded_context": {
            "excluded_row_count": len(excluded_rows),
            "closed_opening_row_count": sum(
                1
                for row in excluded_rows
                if row["exclusion_reason"]
                in {"closed_opening_branch", "closed_opening_subfamily"}
            ),
            "guard_context_row_count": sum(
                1
                for row in excluded_rows
                if row["exclusion_reason"] == "guard_context_only"
            ),
            "reference_exclusion_row_count": sum(
                1
                for row in excluded_rows
                if row["exclusion_reason"]
                in {
                    "reference_unstable",
                    "reference_integrity_error",
                    "train_only_reference",
                }
            ),
        },
        "decision_classification": decision_classification,
        "recommended_next_action": recommended_next_action,
        "inventory_rows": [
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "family_label": row["family_label"],
                "corrected_reference_move": row["corrected_reference_move"],
                "current_selected_move": row["current_artifact_selected_move"],
                "failure_status": row["failure_status"],
                "severity": row["severity"],
                "reference_unstable": row["reference_unstable"],
                "recommended_use": row["recommended_use"],
                "source_report_path": row["source_report_path"],
                "exclusion_reason": row["exclusion_reason"],
                "canonical_state_hash": row["canonical_state_hash"],
            }
            for row in enriched_rows
        ],
        "family_rankings": family_rankings,
        "representative_rows": representative_rows,
        "selected_family": selected_family_report,
    }

    write_json(args.summary_path, summary)
    write_jsonl(args.selected_rows_path, selected_family_rows)
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
