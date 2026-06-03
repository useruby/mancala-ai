#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
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
    OPTIONAL_CHILD_TEACHER_BUDGET,
    ROOT_BUDGETS,
    child_neural_audit,
    child_puct_audit,
    child_state_from_move,
    counterfactual_rows,
    estimate_teacher_5000_budget,
    format_bool,
    format_float,
    load_json,
    load_jsonl,
    load_reference_maps,
    mining_inventory_map,
    python_bin,
    representative_metric_map,
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
DEFAULT_SELECTED_ROWS_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining_v2/"
    "selected_non_opening_family_rows_v2.jsonl"
)
DEFAULT_MINING_SUMMARY_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining_v2/"
    "non_opening_failure_family_summary_v2.json"
)
DEFAULT_SUMMARY_OUT = Path(
    "/tmp/azlite_high_value_swing_value_backup_audit/"
    "high_value_swing_value_backup_audit_summary.json"
)
DEFAULT_REPORT_OUT = Path(
    "docs/alphazero-lite-high-value-swing-value-backup-audit-results.md"
)
FAMILY = "high_value_swing"
SCHEMA = "azlite_high_value_swing_value_backup_audit_v1"
OPTIONAL_ROOT_BUDGET = 2400
MAX_PROJECTED_ROOT_2400_SECONDS = 600.0
REQUIRED_TARGET_ROW_IDS = (
    "high_value_swing-024",
    "high_value_swing-007",
    "high_value_swing-025",
    "high_value_swing-023",
    "high_value_swing-001",
    "high_value_swing-021",
    "high_value_swing-013",
    "high_value_swing-018",
    "high_value_swing-008",
    "high_value_swing-003",
)
REQUIRED_CONTROL_ROW_IDS = (
    "high_value_swing-010",
    "high_value_swing-026",
    "high_value_swing-027",
    "high_value_swing-017",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument(
        "--selected-rows-path", type=Path, default=DEFAULT_SELECTED_ROWS_PATH
    )
    parser.add_argument(
        "--mining-summary-path", type=Path, default=DEFAULT_MINING_SUMMARY_PATH
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpuct", type=float, default=C_PUCT)
    return parser.parse_args(argv)


def rerun_family_mining_if_needed(root: Path, args: argparse.Namespace) -> None:
    if args.selected_rows_path.exists():
        return
    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.run_corrected_non_opening_failure_family_mining_v2",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            "missing selected rows file and rerun failed with exit code "
            f"{completed.returncode}"
        )
    if not args.selected_rows_path.exists():
        raise SystemExit(
            f"selected rows file still missing after rerun: {args.selected_rows_path}"
        )


def policy_rank(root_policy: dict[str, float], move: int) -> int | None:
    ranked = sorted(
        ((int(candidate), float(prob)) for candidate, prob in root_policy.items()),
        key=lambda item: (-item[1], item[0]),
    )
    for index, (candidate, _prob) in enumerate(ranked, start=1):
        if candidate == int(move):
            return index
    return None


def baseline_notes(row: AuditRow, baseline: dict[str, Any]) -> str:
    notes = [row.role]
    if baseline["selected_move"] == row.corrected_reference_move:
        notes.append("selected reference")
    else:
        notes.append("selected away from reference")
    return ", ".join(notes)


def estimate_root_2400_budget(
    *,
    evaluator: ArtifactEvaluator,
    target_rows: list[AuditRow],
    seed: int,
    cpuct: float,
) -> tuple[bool, str]:
    if not target_rows:
        return False, "skipped 2400 root budget: no target rows"
    started = time.perf_counter()
    root_baseline_for_row(
        evaluator=evaluator,
        row=target_rows[0],
        budget=ROOT_BUDGETS[-1],
        seed=int(seed),
        cpuct=float(cpuct),
    )
    elapsed = max(time.perf_counter() - started, 1e-6)
    projected = elapsed * (OPTIONAL_ROOT_BUDGET / ROOT_BUDGETS[-1]) * len(target_rows)
    if projected > MAX_PROJECTED_ROOT_2400_SECONDS:
        return (
            False,
            "skipped 2400 root budget: projected "
            f"~{projected:.1f}s across {len(target_rows)} target rows",
        )
    return (
        True,
        "ran 2400 root budget: projected "
        f"~{projected:.1f}s across {len(target_rows)} target rows",
    )


def validate_and_enrich_rows(
    args: argparse.Namespace,
) -> tuple[list[AuditRow], list[dict[str, Any]]]:
    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_by_id, reference_by_canonical = load_reference_maps(args.reference_path)
    inventory_by_id = mining_inventory_map(args.mining_summary_path)
    representative_by_id = representative_metric_map(args.mining_summary_path)
    selected_rows = load_jsonl(args.selected_rows_path)
    roles_seen: Counter[str] = Counter()
    audit_rows: list[AuditRow] = []
    validation_rows: list[dict[str, Any]] = []
    required_target_set = set(REQUIRED_TARGET_ROW_IDS)
    required_control_set = set(REQUIRED_CONTROL_ROW_IDS)

    for row in selected_rows:
        row_id = str(row["row_id"])
        role = str(row["recommended_role"])
        roles_seen[role] += 1
        if str(row.get("family")) != FAMILY:
            raise ValueError(f"{row_id} family mismatch: expected {FAMILY}")
        suite_row = suite_by_id.get(row_id)
        if suite_row is None:
            raise ValueError(f"{row_id} missing from forensic suite")
        legal_moves = list(suite_row.legal_moves)
        corrected_reference_move = int(row["corrected_reference_move"])
        if corrected_reference_move not in legal_moves:
            raise ValueError(
                f"{row_id} corrected reference move {corrected_reference_move} is illegal"
            )
        canonical = canonical_state_key(suite_row.state)
        if canonical != str(row["canonical_state_hash"]):
            raise ValueError(
                f"{row_id} canonical_state_hash does not match suite canonical state"
            )
        reference_row = reference_by_id.get(row_id) or reference_by_canonical.get(
            canonical
        )
        if reference_row is None:
            raise ValueError(f"{row_id} missing from corrected reference data")
        reference_unstable = bool(reference_row.get("reference_unstable", False))
        if reference_unstable:
            raise ValueError(f"{row_id} is marked reference_unstable")
        inventory_row = inventory_by_id.get(row_id, {})
        failure_status = str(inventory_row.get("failure_status") or "")
        if failure_status == "reference_integrity_error":
            raise ValueError(f"{row_id} is marked reference_integrity_error")
        mining_metrics = representative_by_id.get(row_id, {})
        audit_rows.append(
            AuditRow(
                row_id=row_id,
                role=role,
                severity=str(row.get("severity") or "none"),
                failure_mode=str(row.get("failure_mode") or "unknown"),
                corrected_reference_move=corrected_reference_move,
                current_selected_move=int(row["current_selected_move"]),
                suite_state=dict(suite_row.state),
                legal_moves=legal_moves,
                canonical_state_hash=canonical,
                reference_teacher_value=round_float(reference_row.get("teacher_value")),
                reference_unstable=False,
                reference_integrity_error=False,
                inventory_failure_status=failure_status or None,
                mining_metrics=dict(mining_metrics),
            )
        )
        notes = ["validated"]
        if row_id in required_target_set:
            notes.append("required target row present")
        if row_id in required_control_set:
            notes.append("required control row present")
        validation_rows.append(
            {
                "row_id": row_id,
                "role": role,
                "corrected_reference_move": corrected_reference_move,
                "legal": True,
                "reference_unstable": False,
                "status": "ok",
                "notes": ", ".join(notes),
            }
        )

    for expected_role in (
        "target_candidate",
        "preservation_control",
        "holdout_candidate",
    ):
        if roles_seen[expected_role] == 0:
            raise ValueError(
                f"selected rows are missing required role bucket {expected_role}"
            )
    return audit_rows, validation_rows


def build_root_baseline_record(
    row: AuditRow, baseline: dict[str, Any]
) -> dict[str, Any]:
    return {
        **baseline,
        "selected_is_reference": bool(baseline["pass"]),
        "reference_policy_probability": baseline["root_policy"].get(
            str(row.corrected_reference_move)
        ),
        "selected_policy_probability": None
        if baseline["selected_move"] is None
        else baseline["root_policy"].get(str(int(baseline["selected_move"]))),
        "reference_policy_rank": policy_rank(
            baseline["root_policy"], row.corrected_reference_move
        ),
        "selected_policy_rank": None
        if baseline["selected_move"] is None
        else policy_rank(baseline["root_policy"], int(baseline["selected_move"])),
        "notes": baseline_notes(row, baseline),
    }


def compare_move_consequences(
    *, row: AuditRow, baseline_1200: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    selected_move = baseline_1200["selected_move"]
    baseline_legal_moves = [
        int(move) for move in baseline_1200.get("legal_moves") or []
    ]
    indexed = {
        int(entry["move"]): entry
        for entry in baseline_1200.get("move_consequences") or []
    }
    rows: list[dict[str, Any]] = []
    reference_entry = indexed.get(int(row.corrected_reference_move), {})
    selected_entry = (
        indexed.get(int(selected_move), {}) if selected_move is not None else {}
    )
    notes = []
    if selected_move is not None and selected_move != row.corrected_reference_move:
        if bool(selected_entry.get("gives_extra_turn")) and not bool(
            reference_entry.get("gives_extra_turn")
        ):
            notes.append("selected move gains extra turn")
        if int(selected_entry.get("capture_count", 0)) > int(
            reference_entry.get("capture_count", 0)
        ):
            notes.append("selected move captures more immediately")
        if int(selected_entry.get("immediate_store_delta", 0)) > int(
            reference_entry.get("immediate_store_delta", 0)
        ):
            notes.append("selected move has larger immediate store gain")
        if int(selected_entry.get("remaining_seed_count", 0)) < int(
            reference_entry.get("remaining_seed_count", 0)
        ):
            notes.append("selected move leaves fewer seeds in pits")
    for move in baseline_legal_moves:
        entry = indexed.get(int(move))
        if entry is None:
            raise ValueError(
                f"{row.row_id} missing move consequence for legal move {move}"
            )
        rows.append(
            {
                "row_id": row.row_id,
                "move": int(move),
                "is_corrected_reference": int(move)
                == int(row.corrected_reference_move),
                "is_selected": int(move) == int(selected_move)
                if selected_move is not None
                else False,
                "gives_extra_turn": bool(entry["gives_extra_turn"]),
                "produces_capture": bool(entry["produces_capture"]),
                "capture_count": int(entry["capture_count"]),
                "immediate_store_delta": int(entry["immediate_store_delta"]),
                "side_to_move_after": int(entry["side_to_move_after"]),
                "game_over_after_move": bool(entry["game_over_after_move"]),
                "remaining_seed_count": int(entry["remaining_seed_count"]),
                "store_delta": int(entry["immediate_store_delta"]),
                "pit_index": int(move),
                "move_index": int(move),
                "notes": "; ".join(notes)
                if selected_move is not None
                and int(move) == int(selected_move)
                and notes
                else "",
            }
        )
    return rows, "; ".join(notes) if notes else "no obvious immediate heuristic edge"


def classify_row(
    *,
    row: AuditRow,
    neural_summary: dict[str, Any] | None,
    teacher_budget_summary: dict[int, dict[str, Any]] | None,
    child_puct_budget_summary: dict[int, dict[str, Any]] | None,
    root_baseline_1200: dict[str, Any],
    counterfactual: list[dict[str, Any]],
) -> dict[str, Any]:
    if row.role == "preservation_control":
        return {
            "row_id": row.row_id,
            "role": row.role,
            "row_classification": "inconclusive",
            "supporting_evidence": "preservation control retained to guard against regression",
            "recommended_use": "preserve_control",
            "notes": "baseline pass"
            if root_baseline_1200["pass"]
            else "unexpected control failure",
        }
    if row.role == "holdout_candidate":
        return {
            "row_id": row.row_id,
            "role": row.role,
            "row_classification": "inconclusive",
            "supporting_evidence": "holdout row reserved for out-of-sample follow-up rather than mechanism counting",
            "recommended_use": "holdout",
            "notes": "root still fails"
            if not root_baseline_1200["pass"]
            else "baseline pass",
        }
    assert neural_summary is not None
    assert teacher_budget_summary is not None
    assert child_puct_budget_summary is not None
    deepest_teacher_budget = max(teacher_budget_summary)
    deepest_puct_budget = max(child_puct_budget_summary)
    teacher_prefers_reference = bool(
        teacher_budget_summary[deepest_teacher_budget][
            "teacher_prefers_corrected_reference"
        ]
    )
    teacher_stable = bool(teacher_budget_summary[deepest_teacher_budget]["stable"])
    child_puct_prefers_reference = bool(
        child_puct_budget_summary[deepest_puct_budget][
            "puct_prefers_corrected_reference"
        ]
    )
    neural_prefers_reference = bool(neural_summary["agrees"])
    root_prefers_reference = bool(root_baseline_1200["pass"])
    same_player_ref = int(neural_summary["child_ref_state"]["current_player"]) == int(
        row.suite_state["current_player"]
    )
    same_player_selected = int(
        neural_summary["child_selected_state"]["current_player"]
    ) == int(row.suite_state["current_player"])
    teacher_flip = any(
        entry["intervention"] == "teacher_child_value_override" and entry["flipped"]
        for entry in counterfactual
    )
    prior_flip = any(
        entry["intervention"] == "equalize_root_priors" and entry["flipped"]
        for entry in counterfactual
    )
    swap_flip = any(
        entry["intervention"] == "neural_child_value_swap" and entry["flipped"]
        for entry in counterfactual
    )
    if not teacher_stable:
        classification = "inconclusive"
        evidence = "ClassicMCTS child teacher is unstable across seeds"
    elif not teacher_prefers_reference:
        classification = "corrected_reference_suspicious"
        evidence = (
            "ClassicMCTS child teacher does not support the corrected reference child"
        )
    elif (
        same_player_ref != same_player_selected
        and swap_flip
        and not neural_prefers_reference
    ):
        classification = "backup_perspective_suspect"
        evidence = "sign-sensitive child comparison flips under value swap and perspective differs across children"
    elif teacher_prefers_reference and not neural_prefers_reference:
        classification = "value_head_miscalibration"
        evidence = "ClassicMCTS child teacher prefers corrected reference child but raw neural child values prefer selected child"
    elif (
        teacher_prefers_reference
        and neural_prefers_reference
        and not child_puct_prefers_reference
    ):
        classification = "puct_child_search_value_mismatch"
        evidence = "ClassicMCTS child teacher and neural child values prefer corrected reference child but child PUCT does not"
    elif (
        teacher_prefers_reference
        and neural_prefers_reference
        and child_puct_prefers_reference
        and not root_prefers_reference
    ):
        classification = "root_selection_pressure"
        evidence = "neural, ClassicMCTS child teacher, and child PUCT support corrected reference child but root still selects away"
    elif teacher_flip and not root_prefers_reference:
        classification = "puct_child_search_value_mismatch"
        evidence = "teacher child value override flips the root decision"
    elif prior_flip and not root_prefers_reference:
        classification = "root_selection_pressure"
        evidence = "equalizing root priors flips the root decision"
    elif (
        same_player_ref != same_player_selected
        and not neural_prefers_reference
        and child_puct_prefers_reference
    ):
        classification = "backup_perspective_suspect"
        evidence = (
            "disagreement is concentrated in a perspective-changing child comparison"
        )
    else:
        classification = "inconclusive"
        evidence = "evidence remains mixed after neural, ClassicMCTS, child PUCT, and counterfactual diagnostics"
    return {
        "row_id": row.row_id,
        "role": row.role,
        "row_classification": classification,
        "supporting_evidence": evidence,
        "recommended_use": "target_candidate",
        "notes": "root still fails" if not root_prefers_reference else "baseline pass",
    }


def classify_family(
    row_classifications: list[dict[str, Any]],
) -> tuple[str, str, dict[str, int]]:
    target_rows = [
        row for row in row_classifications if row["role"] == "target_candidate"
    ]
    counts = Counter(str(row["row_classification"]) for row in target_rows)
    dominant = counts.most_common(1)[0][0] if counts else "inconclusive"
    if dominant == "value_head_miscalibration":
        return (
            "value_head_family_gap",
            "build a small train-only child-afterstate value-calibration artifact for high_value_swing, with controls and no arena until local value metrics improve.",
            dict(counts),
        )
    if dominant == "puct_child_search_value_mismatch":
        return (
            "puct_child_search_family_gap",
            "audit child PUCT expansion/backup/value normalization before training.",
            dict(counts),
        )
    if dominant == "root_selection_pressure":
        return (
            "root_selection_family_gap",
            "run cpuct/root-prior calibration diagnostics for high_value_swing.",
            dict(counts),
        )
    if dominant == "corrected_reference_suspicious":
        return (
            "reference_family_uncertain",
            "adjudicate high_value_swing references before training.",
            dict(counts),
        )
    return (
        "mixed_family_gap",
        "split high_value_swing into mechanism-specific buckets and choose the largest stable bucket.",
        dict(counts),
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return output


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite High Value Swing Value/Backup Audit Results",
        "",
        "## 1. Context",
        "",
        "- No training was run.",
        "- No arena was run.",
        "- No model was promoted.",
        "- No replay artifacts were created.",
        "- Corrected references were not mutated.",
        f"- Corrected references: `{summary['inputs']['reference_path']}`.",
        f"- Current artifact: `{summary['inputs']['current_artifact']}`.",
        f"- Selected family rows: `{summary['inputs']['selected_rows_path']}`.",
        "",
        "## 2. Why high_value_swing was selected",
        "",
        "- PR #51 selected `high_value_swing` as the next corrected non-opening family.",
        f"- Family stats: `{json.dumps(summary['family_context'], sort_keys=True)}`.",
        "",
        "## 3. Row validation",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "role",
                "corrected_reference_move",
                "legal",
                "reference_unstable",
                "status",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["role"],
                    str(row["corrected_reference_move"]),
                    format_bool(row["legal"]),
                    format_bool(row["reference_unstable"]),
                    row["status"],
                    row["notes"],
                ]
                for row in summary["row_validation_table"]
            ],
        )
    )
    lines.extend(
        [
            "",
            f"- Perspective conversion: `{summary['perspective_and_backup_check']['conversion_rule']}`.",
            f"- Implementation rule: `{summary['perspective_and_backup_check']['implementation_rule']}`.",
            "",
            "## 4. Root PUCT baseline",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "row_id",
                "role",
                "budget",
                "corrected_reference_move",
                "selected_move",
                "selected_is_reference",
                "reference_visit_share",
                "selected_visit_share",
                "reference_q",
                "selected_q",
                "selected_minus_reference_q_margin",
                "reference_policy_probability",
                "selected_policy_probability",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["role"],
                    str(row["budget"]),
                    str(row["corrected_reference_move"]),
                    str(
                        row["selected_move"]
                        if row["selected_move"] is not None
                        else "-"
                    ),
                    format_bool(row["selected_is_reference"]),
                    format_float(row["reference_visit_share"]),
                    format_float(row["selected_visit_share"]),
                    format_float(row["reference_q"]),
                    format_float(row["selected_q"]),
                    format_float(row["selected_minus_reference_q_margin"]),
                    format_float(row["reference_policy_probability"]),
                    format_float(row["selected_policy_probability"]),
                    row["notes"],
                ]
                for row in summary["root_baseline_table"]
            ],
        )
    )
    lines.extend(["", f"- {summary['root_2400_note']}"])
    lines.extend(["", "## 5. Move consequence audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "move",
                "is_corrected_reference",
                "is_selected",
                "gives_extra_turn",
                "produces_capture",
                "capture_count",
                "immediate_store_delta",
                "side_to_move_after",
                "game_over_after_move",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["move"]),
                    format_bool(row["is_corrected_reference"]),
                    format_bool(row["is_selected"]),
                    format_bool(row["gives_extra_turn"]),
                    format_bool(row["produces_capture"]),
                    str(row["capture_count"]),
                    str(row["immediate_store_delta"]),
                    str(row["side_to_move_after"]),
                    format_bool(row["game_over_after_move"]),
                    row["notes"],
                ]
                for row in summary["move_consequence_table"]
            ],
        )
    )
    lines.extend(["", "## 6. Neural child-afterstate value audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "corrected_reference_move",
                "selected_move",
                "child",
                "raw_value",
                "root_perspective_value",
                "child_ref_minus_child_selected",
                "neural_prefers_reference",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["corrected_reference_move"]),
                    str(row["selected_move"]),
                    row["child"],
                    format_float(row["raw_value"]),
                    format_float(row["root_perspective_value"]),
                    format_float(row["child_ref_minus_child_selected"]),
                    format_bool(row["neural_prefers_reference"]),
                    row["notes"],
                ]
                for row in summary["child_neural_table"]
            ],
        )
    )
    lines.extend(["", "## 7. ClassicMCTS child-afterstate audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "seeds",
                "child_ref_value_root_mean",
                "child_selected_value_root_mean",
                "child_ref_minus_child_selected",
                "teacher_prefers_reference",
                "stable",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    str(row["seeds"]),
                    format_float(row["child_ref_value_root_mean"]),
                    format_float(row["child_selected_value_root_mean"]),
                    format_float(row["child_ref_minus_child_selected"]),
                    format_bool(row["teacher_prefers_reference"]),
                    format_bool(row["stable"]),
                    row["notes"],
                ]
                for row in summary["child_classic_table"]
            ],
        )
    )
    if summary.get("teacher_5000_skip_reason"):
        lines.extend(["", f"- {summary['teacher_5000_skip_reason']}"])
    lines.extend(["", "## 8. PUCT child-afterstate audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "child_ref_value_root",
                "child_selected_value_root",
                "child_ref_minus_child_selected",
                "puct_prefers_reference",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    format_float(row["child_ref_value_root"]),
                    format_float(row["child_selected_value_root"]),
                    format_float(row["child_ref_minus_child_selected"]),
                    format_bool(row["puct_prefers_reference"]),
                    row["notes"],
                ]
                for row in summary["child_puct_table"]
            ],
        )
    )
    lines.extend(["", "## 9. Root counterfactual diagnostics", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "intervention",
                "budget",
                "selected_move",
                "selected_is_reference",
                "reference_visit_share",
                "selected_visit_share",
                "selected_minus_reference_q_margin",
                "flipped",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["intervention"],
                    str(row["budget"]),
                    str(
                        row["selected_move"]
                        if row["selected_move"] is not None
                        else "-"
                    ),
                    format_bool(row["selected_is_corrected_reference"]),
                    format_float(row["reference_visit_share"]),
                    format_float(row["selected_visit_share"]),
                    format_float(row["selected_minus_reference_q_margin"]),
                    format_bool(row["flipped"]),
                    row["notes"],
                ]
                for row in summary["counterfactual_table"]
            ],
        )
    )
    lines.extend(["", "## 10. Row classifications", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "role",
                "row_classification",
                "supporting_evidence",
                "recommended_use",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["role"],
                    row["row_classification"],
                    row["supporting_evidence"],
                    row["recommended_use"],
                    row["notes"],
                ]
                for row in summary["classification_table"]
            ],
        )
    )
    lines.extend(["", "## 11. Family-level interpretation", ""])
    lines.extend(
        markdown_table(
            [
                "family_classification",
                "value_head_miscalibration_count",
                "puct_child_search_mismatch_count",
                "root_selection_pressure_count",
                "corrected_reference_suspicious_count",
                "inconclusive_count",
                "next_action",
            ],
            [
                [
                    summary["family_decision_table"]["family_classification"],
                    str(
                        summary["family_decision_table"][
                            "value_head_miscalibration_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "puct_child_search_mismatch_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "root_selection_pressure_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "corrected_reference_suspicious_count"
                        ]
                    ),
                    str(summary["family_decision_table"]["inconclusive_count"]),
                    summary["family_decision_table"]["next_action"],
                ]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 12. Exactly one recommended next action",
            "",
            f"Recommendation: **{summary['family_decision_table']['next_action']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    rerun_family_mining_if_needed(root, args)
    audit_rows, validation_rows = validate_and_enrich_rows(args)
    evaluator = ArtifactEvaluator(args.current_artifact)
    mining_summary = (
        load_json(args.mining_summary_path) if args.mining_summary_path.exists() else {}
    )
    family_context = next(
        (
            row
            for row in list(mining_summary.get("family_rankings") or [])
            if isinstance(row, dict) and str(row.get("family")) == FAMILY
        ),
        {},
    )

    target_rows = [row for row in audit_rows if row.role == "target_candidate"]
    holdout_rows = [row for row in audit_rows if row.role == "holdout_candidate"]
    control_rows = [row for row in audit_rows if row.role == "preservation_control"]

    run_root_2400, root_2400_note = estimate_root_2400_budget(
        evaluator=evaluator,
        target_rows=target_rows,
        seed=int(args.seed),
        cpuct=float(args.cpuct),
    )
    root_budgets = list(ROOT_BUDGETS)
    if run_root_2400:
        root_budgets.append(OPTIONAL_ROOT_BUDGET)

    root_baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for row in audit_rows:
        for budget in root_budgets:
            root_baselines[(row.row_id, int(budget))] = build_root_baseline_record(
                row,
                root_baseline_for_row(
                    evaluator=evaluator,
                    row=row,
                    budget=int(budget),
                    seed=int(args.seed),
                    cpuct=float(args.cpuct),
                ),
            )

    move_consequence_table: list[dict[str, Any]] = []
    row_move_comparison_notes: dict[str, str] = {}
    for row in audit_rows:
        consequence_rows, note = compare_move_consequences(
            row=row,
            baseline_1200=root_baselines[(row.row_id, 1200)],
        )
        move_consequence_table.extend(consequence_rows)
        row_move_comparison_notes[row.row_id] = note

    teacher_budgets: list[int] = list(CHILD_TEACHER_BASE_BUDGETS)
    teacher_5000_skip_reason = None
    if target_rows:
        can_run_5000, teacher_5000_skip_reason = estimate_teacher_5000_budget(
            target_rows,
            child_state_from_move(
                target_rows[0].suite_state, target_rows[0].corrected_reference_move
            ),
            CHILD_TEACHER_SEEDS,
        )
        if can_run_5000:
            teacher_budgets.append(OPTIONAL_CHILD_TEACHER_BUDGET)

    child_neural_rows: list[dict[str, Any]] = []
    child_classic_rows: list[dict[str, Any]] = []
    child_classic_seed_details: list[dict[str, Any]] = []
    child_puct_rows: list[dict[str, Any]] = []
    counterfactual_table: list[dict[str, Any]] = []
    classification_table: list[dict[str, Any]] = []
    neural_summary_by_row: dict[str, dict[str, Any]] = {}
    teacher_summary_by_row: dict[str, dict[int, dict[str, Any]]] = {}
    child_puct_summary_by_row: dict[str, dict[int, dict[str, Any]]] = {}

    for row in target_rows:
        baseline_1200 = root_baselines[(row.row_id, 1200)]
        neural_rows, neural_summary = child_neural_audit(
            evaluator=evaluator,
            row=row,
            baseline_1200=baseline_1200,
        )
        neural_summary_by_row[row.row_id] = neural_summary
        for neural_row in neural_rows:
            child_neural_rows.append(
                {
                    "row_id": neural_row["row_id"],
                    "corrected_reference_move": neural_row["corrected_reference_move"],
                    "selected_move": neural_row["selected_move"],
                    "child": neural_row["child"],
                    "raw_value": neural_row["raw_value"],
                    "root_perspective_value": neural_row["root_perspective_value"],
                    "child_ref_minus_child_selected": neural_row[
                        "child_ref_minus_child_selected"
                    ],
                    "neural_prefers_reference": neural_row[
                        "agrees_with_corrected_reference"
                    ],
                    "child_legal_moves": neural_row["child_legal_moves"],
                    "child_policy_top_move": neural_row["child_policy_top_move"],
                    "notes": neural_row["notes"],
                }
            )
        teacher_rows, teacher_budget_summary = teacher_child_audit(
            row=row,
            child_summary=neural_summary,
            budgets=tuple(teacher_budgets),
            seeds=CHILD_TEACHER_SEEDS,
        )
        child_classic_seed_details.extend(teacher_rows)
        teacher_summary_by_row[row.row_id] = teacher_budget_summary
        for budget in teacher_budgets:
            aggregate = teacher_budget_summary[int(budget)]
            child_classic_rows.append(
                {
                    "row_id": row.row_id,
                    "budget": int(budget),
                    "seeds": list(CHILD_TEACHER_SEEDS),
                    "child_ref_value_root_mean": aggregate["mean_child_ref_value_root"],
                    "child_selected_value_root_mean": aggregate[
                        "mean_child_selected_value_root"
                    ],
                    "child_ref_minus_child_selected": aggregate["mean_diff"],
                    "teacher_prefers_reference": aggregate[
                        "teacher_prefers_corrected_reference"
                    ],
                    "stable": aggregate["stable"],
                    "notes": "ClassicMCTS child-afterstate teacher aggregate",
                }
            )
        puct_rows, puct_budget_summary = child_puct_audit(
            evaluator=evaluator,
            row=row,
            child_summary=neural_summary,
            budgets=CHILD_PUCT_BUDGETS,
            seed=int(args.seed),
            cpuct=float(args.cpuct),
        )
        child_puct_summary_by_row[row.row_id] = puct_budget_summary
        for puct_row in puct_rows:
            child_puct_rows.append(
                {
                    **puct_row,
                    "puct_prefers_reference": puct_row[
                        "puct_prefers_corrected_reference"
                    ],
                }
            )
        counterfactual_table.extend(
            counterfactual_rows(
                evaluator=evaluator,
                row=row,
                child_summary=neural_summary,
                teacher_budget_summary=teacher_budget_summary,
                neural_summary=neural_summary,
                seed=int(args.seed),
                cpuct=float(args.cpuct),
            )
        )

    for row in audit_rows:
        classification_table.append(
            classify_row(
                row=row,
                neural_summary=neural_summary_by_row.get(row.row_id),
                teacher_budget_summary=teacher_summary_by_row.get(row.row_id),
                child_puct_budget_summary=child_puct_summary_by_row.get(row.row_id),
                root_baseline_1200=root_baselines[(row.row_id, 1200)],
                counterfactual=[
                    entry
                    for entry in counterfactual_table
                    if entry["row_id"] == row.row_id
                ],
            )
        )

    family_label, family_action, family_counts = classify_family(classification_table)
    root_baseline_table = [
        root_baselines[key]
        for key in sorted(root_baselines, key=lambda item: (item[0], item[1]))
    ]
    family_decision_table = {
        "family_classification": family_label,
        "value_head_miscalibration_count": int(
            family_counts.get("value_head_miscalibration", 0)
        ),
        "puct_child_search_mismatch_count": int(
            family_counts.get("puct_child_search_value_mismatch", 0)
        ),
        "root_selection_pressure_count": int(
            family_counts.get("root_selection_pressure", 0)
        ),
        "corrected_reference_suspicious_count": int(
            family_counts.get("corrected_reference_suspicious", 0)
        ),
        "inconclusive_count": int(family_counts.get("inconclusive", 0)),
        "next_action": family_action,
    }
    summary = {
        "schema": SCHEMA,
        "selected_family": FAMILY,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "selected_rows_path": str(args.selected_rows_path),
            "mining_summary_path": str(args.mining_summary_path),
        },
        "guardrails": {
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
            "mutated_corrected_references": False,
        },
        "family_context": family_context,
        "row_selection": {
            "target_row_ids": [row.row_id for row in target_rows],
            "holdout_row_ids": [row.row_id for row in holdout_rows],
            "control_row_ids": [row.row_id for row in control_rows],
            "required_target_rows_present": [
                row_id
                for row_id in REQUIRED_TARGET_ROW_IDS
                if row_id in {row.row_id for row in audit_rows}
            ],
            "required_control_rows_present": [
                row_id
                for row_id in REQUIRED_CONTROL_ROW_IDS
                if row_id in {row.row_id for row in audit_rows}
            ],
        },
        "perspective_and_backup_check": {
            "conversion_rule": "+1 when child current_player == root current_player, else -1",
            "implementation_rule": "PUCT _search negates the returned child value exactly when child.game.current_player != parent.game.current_player",
            "notes": "extra turns keep sign; turn handoff flips sign back to root-player perspective",
        },
        "root_2400_note": root_2400_note,
        "teacher_5000_skip_reason": teacher_5000_skip_reason,
        "row_validation_table": validation_rows,
        "root_baseline_table": root_baseline_table,
        "move_consequence_table": move_consequence_table,
        "row_move_comparison_notes": row_move_comparison_notes,
        "child_neural_table": child_neural_rows,
        "child_classic_table": child_classic_rows,
        "child_classic_seed_details": child_classic_seed_details,
        "child_puct_table": child_puct_rows,
        "counterfactual_table": counterfactual_table,
        "classification_table": classification_table,
        "family_decision_table": family_decision_table,
    }
    write_json(args.summary_out, summary)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.summary_out),
                "report_path": str(args.report_out),
                "family_classification": family_label,
                "recommended_next_action": family_action,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
