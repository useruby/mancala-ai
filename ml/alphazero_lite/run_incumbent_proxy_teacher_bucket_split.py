#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    load_json,
    round_float,
    write_json,
)
from ml.alphazero_lite.run_incumbent_proxy_teacher_policy_decision_audit import (
    canonical_reference_rows,
    deterministic_puct_run,
)


DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_DECISION_SUMMARY_PATH = Path(
    "/tmp/azlite_incumbent_proxy_teacher_policy_decision/"
    "teacher_policy_decision_summary.json"
)
DEFAULT_DECISION_REPORT_PATH = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-teacher-policy-decision-results.md"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/azlite_incumbent_proxy_teacher_bucket_split")
DEFAULT_SUMMARY_OUT = DEFAULT_OUTPUT_DIR / "teacher_bucket_split_summary.json"
DEFAULT_CLASSIC_BUCKET_OUT = DEFAULT_OUTPUT_DIR / "classic_teacher_bucket.json"
DEFAULT_PUCT_BUCKET_OUT = DEFAULT_OUTPUT_DIR / "puct_teacher_bucket.json"
DEFAULT_EXCLUDED_BUCKET_OUT = DEFAULT_OUTPUT_DIR / "excluded_diagnostic_bucket.json"
DEFAULT_CLASSIC_ROWS_OUT = DEFAULT_OUTPUT_DIR / "classic_teacher_rows.jsonl"
DEFAULT_PUCT_ROWS_OUT = DEFAULT_OUTPUT_DIR / "puct_teacher_rows.jsonl"
DEFAULT_EXCLUDED_ROWS_OUT = DEFAULT_OUTPUT_DIR / "excluded_diagnostic_rows.jsonl"
DEFAULT_REPORT_OUT = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-teacher-bucket-split-results.md"
)

SCHEMA = "azlite_incumbent_proxy_teacher_bucket_split_v1"
FAMILY = "incumbent_proxy_disagreement"
CLASSIC_BUCKET = "classic_teacher"
PUCT_BUCKET = "puct_teacher"
EXCLUDED_BUCKET = "excluded_diagnostic"
CLASSIC_ROW_IDS = (
    "incumbent_proxy_disagreement-008",
    "incumbent_proxy_disagreement-014",
    "incumbent_proxy_disagreement-022",
    "incumbent_proxy_disagreement-024",
    "incumbent_proxy_disagreement-025",
    "incumbent_proxy_disagreement-035",
)
PUCT_ROW_IDS = (
    "incumbent_proxy_disagreement-007",
    "incumbent_proxy_disagreement-009",
    "incumbent_proxy_disagreement-021",
    "incumbent_proxy_disagreement-026",
    "incumbent_proxy_disagreement-028",
    "incumbent_proxy_disagreement-032",
    "incumbent_proxy_disagreement-033",
)
EXCLUDED_ROW_IDS = (
    "incumbent_proxy_disagreement-003",
    "incumbent_proxy_disagreement-010",
    "incumbent_proxy_disagreement-012",
    "incumbent_proxy_disagreement-018",
    "incumbent_proxy_disagreement-020",
    "incumbent_proxy_disagreement-023",
    "incumbent_proxy_disagreement-027",
    "incumbent_proxy_disagreement-029",
)
INITIAL_BUCKET_BY_ROW = {
    **{row_id: CLASSIC_BUCKET for row_id in CLASSIC_ROW_IDS},
    **{row_id: PUCT_BUCKET for row_id in PUCT_ROW_IDS},
    **{row_id: EXCLUDED_BUCKET for row_id in EXCLUDED_ROW_IDS},
}
EXPECTED_ROW_IDS = tuple(sorted(INITIAL_BUCKET_BY_ROW))
CURRENT_BUDGETS = (384, 1200)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument(
        "--decision-summary-path", type=Path, default=DEFAULT_DECISION_SUMMARY_PATH
    )
    parser.add_argument(
        "--decision-report-path", type=Path, default=DEFAULT_DECISION_REPORT_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument(
        "--classic-bucket-out", type=Path, default=DEFAULT_CLASSIC_BUCKET_OUT
    )
    parser.add_argument("--puct-bucket-out", type=Path, default=DEFAULT_PUCT_BUCKET_OUT)
    parser.add_argument(
        "--excluded-bucket-out", type=Path, default=DEFAULT_EXCLUDED_BUCKET_OUT
    )
    parser.add_argument(
        "--classic-rows-out", type=Path, default=DEFAULT_CLASSIC_ROWS_OUT
    )
    parser.add_argument("--puct-rows-out", type=Path, default=DEFAULT_PUCT_ROWS_OUT)
    parser.add_argument(
        "--excluded-rows-out", type=Path, default=DEFAULT_EXCLUDED_ROWS_OUT
    )
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    return parser.parse_args(argv)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def parse_markdown_row_decisions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    section = None
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## 8. Row-level teacher decisions"):
            section = "row_decisions"
            continue
        if line.startswith("## "):
            section = None
            continue
        if section != "row_decisions":
            continue
        if not line.startswith("| incumbent_proxy_disagreement-"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 7:
            continue
        rows.append(
            {
                "row_id": parts[0],
                "row_decision": parts[1],
                "preferred_teacher": parts[2],
                "preferred_move": int(parts[3]),
                "active_reference_move": int(parts[4]),
                "recommended_use": parts[5],
                "evidence_summary": parts[6],
            }
        )
    return rows


def load_teacher_decisions(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    if args.decision_summary_path.exists():
        payload = load_json(args.decision_summary_path)
        rows = list(payload.get("row_decision_table") or [])
        if rows:
            return rows, str(args.decision_summary_path)
    rows = parse_markdown_row_decisions(args.decision_report_path)
    return rows, str(args.decision_report_path)


def expected_bucket_for_row(row_id: str) -> str:
    bucket = INITIAL_BUCKET_BY_ROW.get(row_id)
    if bucket is None:
        raise ValueError(f"unexpected row id {row_id}")
    return bucket


def excluded_reason(row_id: str, source_row: dict[str, Any], status: str) -> str:
    if status == "reference_integrity_error":
        return "reference_integrity_error"
    if status == "puct_preference_not_reproduced":
        return "puct_preference_not_reproduced"
    if row_id == "incumbent_proxy_disagreement-010":
        return "excluded_unstable"
    if source_row["row_decision"] == "unstable_or_inconclusive":
        if "mixed" in str(source_row.get("evidence_summary") or ""):
            return "teacher_evidence_mixed"
        return "unstable_or_inconclusive"
    return "unstable_or_inconclusive"


def classic_recommended_role(row: dict[str, Any]) -> str:
    if bool(row["pass_384"]) and bool(row["pass_1200"]):
        return "preservation_control"
    if row["current_selected_1200"] == row["active_reference_move"]:
        return "diagnostic_control"
    return "target_candidate"


def duplicate_counts(rows: list[dict[str, Any]]) -> tuple[int, dict[str, int]]:
    counts = Counter(str(row["canonical_state_hash"]) for row in rows)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    return duplicate_count, dict(counts)


def choose_final_recommendation(
    bucket_summary_table: list[dict[str, Any]],
) -> dict[str, str]:
    by_bucket = {row["bucket"]: row for row in bucket_summary_table}
    classic = by_bucket[CLASSIC_BUCKET]
    puct = by_bucket[PUCT_BUCKET]
    excluded = by_bucket[EXCLUDED_BUCKET]
    classic_targets = int(classic["train_target_eligible_count"])
    classic_controls = int(classic["preservation_control_count"])
    classic_errors = int(classic["integrity_errors"])
    puct_rows = int(puct["row_count"])
    puct_errors = int(puct["integrity_errors"])
    excluded_rows = int(excluded["row_count"])

    if classic_errors > 0 or puct_errors > 0:
        classification = "bucket_split_not_ready"
        supporting_evidence = f"validation found integrity errors: classic={classic_errors}, puct={puct_errors}"
        rejected_alternatives = "rejected classic and PUCT follow-up branches until reference/state integrity is clean"
        next_action = (
            "fix reference/state integrity or rerun teacher-policy decision audit."
        )
    elif classic_targets == 0:
        classification = "no_classic_training_signal"
        supporting_evidence = f"Classic bucket has {classic_controls} preservation controls and no failing target rows"
        rejected_alternatives = "rejected Classic diagnostic artifact because the bucket does not contain a live training signal"
        next_action = "pursue PUCT-teacher reference artifact or mine the next non-opening family."
    elif classic_targets >= 2 and classic_controls >= 1 and puct_rows > 0:
        classification = "split_teacher_policy_required"
        supporting_evidence = (
            f"Classic bucket has {classic_targets} stable target candidates plus {classic_controls} controls, "
            f"while PUCT still has {puct_rows} reproduced rows that require a separate reference-policy branch; "
            f"excluded diagnostic bucket remains isolated at {excluded_rows} rows"
        )
        rejected_alternatives = (
            "rejected a single Classic-only branch because useful PUCT-preferred rows remain incompatible; "
            "rejected a PUCT-first branch because it would require a separate reference artifact before any safe training target exists"
        )
        next_action = "keep separate Classic-target and PUCT-reference branches; run Classic diagnostic artifact first because it does not require mutating references."
    elif puct_rows > int(classic["row_count"]):
        classification = "puct_teacher_reference_artifact_needed"
        supporting_evidence = f"PUCT bucket remains larger/cleaner than Classic: puct_rows={puct_rows}, classic_rows={classic['row_count']}"
        rejected_alternatives = "rejected Classic-first follow-up because the remaining clean signal is stronger under the PUCT teacher"
        next_action = "create a non-mutating PUCT-teacher reference artifact and rerun corrected rebaseline under that alternate teacher policy."
    elif classic_targets >= 2 and classic_controls >= 1:
        classification = "classic_teacher_bucket_ready_for_diagnostic_artifact"
        supporting_evidence = f"Classic bucket has {classic_targets} stable target candidates plus {classic_controls} controls"
        rejected_alternatives = "rejected PUCT-first follow-up because the Classic branch is already sufficient for a non-mutating diagnostic artifact"
        next_action = "build a tiny Classic-teacher diagnostic artifact from only Classic-confirmed target rows plus controls; no arena until local metrics improve."
    else:
        classification = "bucket_split_not_ready"
        supporting_evidence = (
            "validation did not produce a clean, targetable next branch"
        )
        rejected_alternatives = "rejected training follow-up because the bucket split still lacks a safe target set"
        next_action = (
            "fix reference/state integrity or rerun teacher-policy decision audit."
        )

    return {
        "classification": classification,
        "supporting_evidence": supporting_evidence,
        "rejected_alternatives": rejected_alternatives,
        "next_action": next_action,
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Incumbent Proxy Disagreement Teacher Bucket Split Results",
        "",
        "## 1. Context",
        "",
        "- No training was run.",
        "- No arena was run.",
        "- No model was promoted.",
        "- Active references stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.",
        "- No production replay artifact was created.",
        f"- Current artifact: `{summary['inputs']['current_artifact']}`.",
        f"- Teacher decisions were loaded from `{summary['decision_source']}`.",
        "",
        "## 2. Why PR #46 blocked training",
        "",
        "- PR #46 concluded that this family required a teacher-policy split before any training target could be constructed.",
        f"- Family decision stayed `{summary['source_family_decision']['family_decision']}` with counts Classic=`{summary['source_family_decision']['classic_reference_confirmed_count']}`, PUCT=`{summary['source_family_decision']['puct_reference_preferred_count']}`, excluded=`{summary['source_family_decision']['unstable_or_inconclusive_count']}`.",
        "- Mixing Classic-teacher and PUCT-teacher rows into one target would still blend incompatible labels.",
        "",
        "## 3. Bucket definitions",
        "",
        f"- Classic-teacher bucket: `{', '.join(CLASSIC_ROW_IDS)}`.",
        f"- PUCT-teacher bucket: `{', '.join(PUCT_ROW_IDS)}`.",
        f"- Excluded diagnostic bucket: `{', '.join(EXCLUDED_ROW_IDS)}`.",
        "- Any validation or integrity failure is forced into the excluded diagnostic bucket.",
        "",
        "## 4. Reference and state validation",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "bucket",
                "canonical_state_match",
                "active_reference_legal",
                "preferred_move_legal",
                "duplicate_state_conflict",
                "status",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["bucket"],
                    format_bool(row["canonical_state_match"]),
                    format_bool(row["active_reference_legal"]),
                    format_bool(row["preferred_move_legal"]),
                    format_bool(row["duplicate_state_conflict"]),
                    row["status"],
                    row["notes"],
                ]
                for row in summary["validation_table"]
            ],
        )
    )
    lines.extend(["", "## 5. Classic-teacher bucket", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "active_reference_move",
                "current_selected_384",
                "current_selected_1200",
                "pass_384",
                "pass_1200",
                "reference_visit_share_384",
                "reference_visit_share_1200",
                "recommended_role",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["active_reference_move"]),
                    str(row["current_selected_384"]),
                    str(row["current_selected_1200"]),
                    format_bool(row["pass_384"]),
                    format_bool(row["pass_1200"]),
                    format_float(row["reference_visit_share_384"]),
                    format_float(row["reference_visit_share_1200"]),
                    row["recommended_role"],
                    row["notes"],
                ]
                for row in summary["classic_bucket_table"]
            ],
        )
    )
    lines.extend(["", "## 6. PUCT-teacher bucket", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "active_reference_move",
                "puct_preferred_move",
                "current_selected_384",
                "current_selected_1200",
                "puct_preference_still_reproduced",
                "recommended_use",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["active_reference_move"]),
                    str(row["preferred_move"]),
                    str(row["current_selected_384"]),
                    str(row["current_selected_1200"]),
                    format_bool(row["puct_preference_still_reproduced"]),
                    row["recommended_use"],
                    row["notes"],
                ]
                for row in summary["puct_bucket_table"]
            ],
        )
    )
    lines.extend(["", "## 7. Excluded diagnostic bucket", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "bucket",
                "row_decision",
                "preferred_teacher",
                "preferred_move",
                "active_reference_move",
                "recommended_role",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["bucket"],
                    row["row_decision"],
                    row["preferred_teacher"],
                    str(row["preferred_move"]),
                    str(row["active_reference_move"]),
                    row["recommended_role"],
                    row["notes"],
                ]
                for row in summary["excluded_bucket_table"]
            ],
        )
    )
    lines.extend(["", "## 8. Lightweight bucket validation", ""])
    lines.extend(
        markdown_table(
            [
                "bucket",
                "row_count",
                "train_target_eligible_count",
                "preservation_control_count",
                "excluded_count",
                "integrity_errors",
                "targetability",
                "next_action",
            ],
            [
                [
                    row["bucket"],
                    str(row["row_count"]),
                    str(row["train_target_eligible_count"]),
                    str(row["preservation_control_count"]),
                    str(row["excluded_count"]),
                    str(row["integrity_errors"]),
                    row["targetability"],
                    row["next_action"],
                ]
                for row in summary["bucket_summary_table"]
            ],
        )
    )
    lines.extend(["", "## 9. Targetability decision", ""])
    lines.extend(
        markdown_table(
            [
                "classification",
                "supporting_evidence",
                "rejected_alternatives",
                "next_action",
            ],
            [
                [
                    summary["decision_table"]["classification"],
                    summary["decision_table"]["supporting_evidence"],
                    summary["decision_table"]["rejected_alternatives"],
                    summary["decision_table"]["next_action"],
                ]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 10. Exactly one recommended next action",
            "",
            f"Recommendation: **{summary['decision_table']['next_action']}**",
            "",
        ]
    )
    lines.extend(["## Bucket Assignment Table", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "bucket",
                "row_decision",
                "preferred_teacher",
                "preferred_move",
                "active_reference_move",
                "recommended_role",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["bucket"],
                    row["row_decision"],
                    row["preferred_teacher"],
                    str(row["preferred_move"]),
                    str(row["active_reference_move"]),
                    row["recommended_role"],
                    row["notes"],
                ]
                for row in summary["bucket_assignment_table"]
            ],
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decision_rows, decision_source = load_teacher_decisions(args)
    decision_by_id = {
        str(row["row_id"]): dict(row)
        for row in decision_rows
        if str(row.get("row_id", "")).startswith("incumbent_proxy_disagreement-")
    }
    missing = sorted(set(EXPECTED_ROW_IDS) - set(decision_by_id))
    if missing:
        raise ValueError(f"missing teacher decisions for rows: {missing}")

    source_family_decision = {
        "family_decision": "teacher_policy_split_required",
        "classic_reference_confirmed_count": 6,
        "puct_reference_preferred_count": 7,
        "unstable_or_inconclusive_count": 8,
    }
    if args.decision_summary_path.exists():
        decision_summary = load_json(args.decision_summary_path)
        source_family_decision = dict(
            decision_summary.get("family_decision_table") or source_family_decision
        )

    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_by_id = canonical_reference_rows(load_json(args.reference_path))
    evaluator = ArtifactEvaluator(args.current_artifact)

    provisional_rows: list[dict[str, Any]] = []
    assigned_bucket_ids: dict[str, list[str]] = defaultdict(list)
    for row_id in EXPECTED_ROW_IDS:
        source_row = decision_by_id[row_id]
        suite_row = suite_by_id.get(row_id)
        reference_row = reference_by_id.get(row_id)
        if suite_row is None or reference_row is None:
            raise ValueError(f"missing suite/reference row for {row_id}")

        state = dict(suite_row.state)
        reference_state = dict(reference_row["state"])
        canonical_hash = canonical_state_key(state)
        canonical_match = canonical_hash == canonical_state_key(reference_state)
        legal_moves = [int(move) for move in suite_row.legal_moves]
        active_reference_move = int(reference_row["reference_move"])
        preferred_move = int(source_row["preferred_move"])
        current_384 = deterministic_puct_run(evaluator, state, budget=384)
        current_1200 = deterministic_puct_run(evaluator, state, budget=1200)
        active_reference_legal = active_reference_move in legal_moves
        preferred_move_legal = preferred_move in legal_moves
        status = "ok"
        notes: list[str] = []
        if not canonical_match:
            status = "reference_integrity_error"
            notes.append("canonical state hash mismatch")
        if not active_reference_legal:
            status = "reference_integrity_error"
            notes.append("active reference move is illegal")
        if not preferred_move_legal:
            status = "reference_integrity_error"
            notes.append("preferred move is illegal")
        if row_id == "incumbent_proxy_disagreement-010":
            notes.append("row 010 remains excluded")

        final_bucket = expected_bucket_for_row(row_id)
        if (
            final_bucket == PUCT_BUCKET
            and current_1200["selected_move"] != preferred_move
        ):
            final_bucket = EXCLUDED_BUCKET
            if status == "ok":
                status = "puct_preference_not_reproduced"
            notes.append(
                "deterministic 1200 no longer reproduces the audited PUCT preference"
            )
        if status == "reference_integrity_error":
            final_bucket = EXCLUDED_BUCKET

        pass_384 = current_384["selected_move"] == active_reference_move
        pass_1200 = current_1200["selected_move"] == active_reference_move
        current_selected_384 = current_384["selected_move"]
        current_selected_1200 = current_1200["selected_move"]

        if final_bucket == CLASSIC_BUCKET:
            recommended_role = classic_recommended_role(
                {
                    "pass_384": pass_384,
                    "pass_1200": pass_1200,
                    "current_selected_1200": current_selected_1200,
                    "active_reference_move": active_reference_move,
                }
            )
        elif final_bucket == PUCT_BUCKET:
            recommended_role = "reference_policy_decision_only"
        else:
            recommended_role = "excluded_diagnostic"

        row_payload = {
            "row_id": row_id,
            "bucket": final_bucket,
            "initial_bucket": expected_bucket_for_row(row_id),
            "row_decision": str(source_row["row_decision"]),
            "preferred_teacher": str(source_row["preferred_teacher"]),
            "preferred_move": preferred_move,
            "active_reference_move": active_reference_move,
            "recommended_use": str(source_row["recommended_use"]),
            "evidence_summary": str(source_row["evidence_summary"]),
            "recommended_role": recommended_role,
            "suite_state": state,
            "canonical_state_hash": canonical_hash,
            "canonical_state_match": canonical_match,
            "legal_moves": legal_moves,
            "active_reference_legal": active_reference_legal,
            "preferred_move_legal": preferred_move_legal,
            "current_selected_move": current_selected_1200,
            "current_selected_384": current_selected_384,
            "current_selected_1200": current_selected_1200,
            "pass_384": pass_384,
            "pass_1200": pass_1200,
            "reference_visit_share_384": round_float(
                current_384["visit_share_by_move"].get(active_reference_move)
            ),
            "reference_visit_share_1200": round_float(
                current_1200["visit_share_by_move"].get(active_reference_move)
            ),
            "puct_preference_still_reproduced": current_selected_1200 == preferred_move,
            "active_reference_should_not_be_used_for_training": final_bucket
            == PUCT_BUCKET,
            "requires_reference_policy_decision": final_bucket == PUCT_BUCKET,
            "do_not_train": final_bucket in {PUCT_BUCKET, EXCLUDED_BUCKET},
            "do_not_gate": final_bucket == EXCLUDED_BUCKET,
            "status": status,
            "reason": excluded_reason(row_id, source_row, status)
            if final_bucket == EXCLUDED_BUCKET
            else None,
            "notes": "; ".join(notes)
            if notes
            else "validated against active references",
        }
        provisional_rows.append(row_payload)
        assigned_bucket_ids[row_id].append(final_bucket)

    duplicate_row_ids = {
        row_id: buckets
        for row_id, buckets in assigned_bucket_ids.items()
        if len(buckets) > 1
    }
    if duplicate_row_ids:
        raise ValueError(f"rows assigned to multiple buckets: {duplicate_row_ids}")

    canonical_buckets: dict[str, set[str]] = defaultdict(set)
    for row in provisional_rows:
        canonical_buckets[str(row["canonical_state_hash"])].add(str(row["bucket"]))
    for row in provisional_rows:
        row["duplicate_state_conflict"] = (
            len(canonical_buckets[str(row["canonical_state_hash"])]) > 1
        )

    bucket_rows = {
        CLASSIC_BUCKET: sorted(
            [row for row in provisional_rows if row["bucket"] == CLASSIC_BUCKET],
            key=lambda row: row["row_id"],
        ),
        PUCT_BUCKET: sorted(
            [row for row in provisional_rows if row["bucket"] == PUCT_BUCKET],
            key=lambda row: row["row_id"],
        ),
        EXCLUDED_BUCKET: sorted(
            [row for row in provisional_rows if row["bucket"] == EXCLUDED_BUCKET],
            key=lambda row: row["row_id"],
        ),
    }

    bucket_summary_table: list[dict[str, Any]] = []
    for bucket_name in (CLASSIC_BUCKET, PUCT_BUCKET, EXCLUDED_BUCKET):
        rows = bucket_rows[bucket_name]
        duplicate_state_count, _counts = duplicate_counts(rows)
        integrity_errors = sum(
            1 for row in rows if row["status"] == "reference_integrity_error"
        )
        preservation_control_count = sum(
            1 for row in rows if row["recommended_role"] == "preservation_control"
        )
        train_target_eligible_count = sum(
            1 for row in rows if row["recommended_role"] == "target_candidate"
        )
        excluded_count = len(rows) if bucket_name == EXCLUDED_BUCKET else 0
        bucket_conflicts = sum(1 for row in rows if row["duplicate_state_conflict"])
        if bucket_name == CLASSIC_BUCKET:
            targetability = (
                "classic_diagnostic_ready"
                if train_target_eligible_count >= 2
                else "controls_only"
            )
            next_action = (
                "use target candidates plus controls for a tiny Classic-teacher diagnostic artifact"
                if train_target_eligible_count >= 2
                else "keep as preservation-only controls"
            )
        elif bucket_name == PUCT_BUCKET:
            targetability = "reference_policy_only"
            next_action = "use only for a future separate PUCT-teacher reference artifact decision"
        else:
            targetability = "not_targetable"
            next_action = "keep diagnostic-only and exclude from training and gates"
        bucket_summary_table.append(
            {
                "bucket": bucket_name,
                "row_count": len(rows),
                "legal_reference_count": sum(
                    1 for row in rows if row["active_reference_legal"]
                ),
                "legal_preferred_move_count": sum(
                    1 for row in rows if row["preferred_move_legal"]
                ),
                "reference_integrity_errors": integrity_errors,
                "duplicate_state_count": duplicate_state_count,
                "bucket_conflicts": bucket_conflicts,
                "train_target_eligible_count": train_target_eligible_count,
                "preservation_control_count": preservation_control_count,
                "excluded_count": excluded_count,
                "integrity_errors": integrity_errors,
                "targetability": targetability,
                "next_action": next_action,
            }
        )

    decision_table = choose_final_recommendation(bucket_summary_table)

    classic_bucket_table = [
        {
            "row_id": row["row_id"],
            "active_reference_move": row["active_reference_move"],
            "current_selected_384": row["current_selected_384"],
            "current_selected_1200": row["current_selected_1200"],
            "pass_384": row["pass_384"],
            "pass_1200": row["pass_1200"],
            "reference_visit_share_384": row["reference_visit_share_384"],
            "reference_visit_share_1200": row["reference_visit_share_1200"],
            "recommended_role": row["recommended_role"],
            "notes": row["evidence_summary"],
        }
        for row in bucket_rows[CLASSIC_BUCKET]
    ]
    puct_bucket_table = [
        {
            "row_id": row["row_id"],
            "active_reference_move": row["active_reference_move"],
            "preferred_move": row["preferred_move"],
            "current_selected_384": row["current_selected_384"],
            "current_selected_1200": row["current_selected_1200"],
            "puct_preference_still_reproduced": row["puct_preference_still_reproduced"],
            "recommended_use": row["recommended_use"],
            "notes": row["evidence_summary"],
        }
        for row in bucket_rows[PUCT_BUCKET]
    ]
    excluded_bucket_table = [
        {
            "row_id": row["row_id"],
            "bucket": row["bucket"],
            "row_decision": row["row_decision"],
            "preferred_teacher": row["preferred_teacher"],
            "preferred_move": row["preferred_move"],
            "active_reference_move": row["active_reference_move"],
            "recommended_role": row["recommended_role"],
            "notes": row["reason"] or row["notes"],
        }
        for row in bucket_rows[EXCLUDED_BUCKET]
    ]

    bucket_assignment_table = [
        {
            "row_id": row["row_id"],
            "bucket": row["bucket"],
            "row_decision": row["row_decision"],
            "preferred_teacher": row["preferred_teacher"],
            "preferred_move": row["preferred_move"],
            "active_reference_move": row["active_reference_move"],
            "recommended_role": row["recommended_role"],
            "notes": row["evidence_summary"],
        }
        for row in sorted(provisional_rows, key=lambda row: row["row_id"])
    ]
    validation_table = [
        {
            "row_id": row["row_id"],
            "bucket": row["bucket"],
            "canonical_state_match": row["canonical_state_match"],
            "active_reference_legal": row["active_reference_legal"],
            "preferred_move_legal": row["preferred_move_legal"],
            "duplicate_state_conflict": row["duplicate_state_conflict"],
            "status": row["status"],
            "notes": row["notes"],
        }
        for row in sorted(provisional_rows, key=lambda row: row["row_id"])
    ]

    classic_bucket_artifact = {
        "schema": SCHEMA,
        "family": FAMILY,
        "bucket": CLASSIC_BUCKET,
        "row_count": len(bucket_rows[CLASSIC_BUCKET]),
        "rows": bucket_rows[CLASSIC_BUCKET],
    }
    puct_bucket_artifact = {
        "schema": SCHEMA,
        "family": FAMILY,
        "bucket": PUCT_BUCKET,
        "row_count": len(bucket_rows[PUCT_BUCKET]),
        "rows": bucket_rows[PUCT_BUCKET],
    }
    excluded_bucket_artifact = {
        "schema": SCHEMA,
        "family": FAMILY,
        "bucket": EXCLUDED_BUCKET,
        "row_count": len(bucket_rows[EXCLUDED_BUCKET]),
        "rows": bucket_rows[EXCLUDED_BUCKET],
    }

    summary = {
        "schema": SCHEMA,
        "family": FAMILY,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "decision_summary_path": str(args.decision_summary_path),
            "decision_report_path": str(args.decision_report_path),
            "output_dir": str(args.output_dir),
        },
        "decision_source": decision_source,
        "source_family_decision": source_family_decision,
        "validation_table": validation_table,
        "bucket_assignment_table": bucket_assignment_table,
        "classic_bucket_table": classic_bucket_table,
        "puct_bucket_table": puct_bucket_table,
        "excluded_bucket_table": excluded_bucket_table,
        "bucket_summary_table": bucket_summary_table,
        "decision_table": decision_table,
        "classic_teacher_bucket": classic_bucket_artifact,
        "puct_teacher_bucket": puct_bucket_artifact,
        "excluded_diagnostic_bucket": excluded_bucket_artifact,
    }

    write_json(args.summary_out, summary)
    write_json(args.classic_bucket_out, classic_bucket_artifact)
    write_json(args.puct_bucket_out, puct_bucket_artifact)
    write_json(args.excluded_bucket_out, excluded_bucket_artifact)
    write_jsonl(args.classic_rows_out, bucket_rows[CLASSIC_BUCKET])
    write_jsonl(args.puct_rows_out, bucket_rows[PUCT_BUCKET])
    write_jsonl(args.excluded_rows_out, bucket_rows[EXCLUDED_BUCKET])
    args.report_out.write_text(build_report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
