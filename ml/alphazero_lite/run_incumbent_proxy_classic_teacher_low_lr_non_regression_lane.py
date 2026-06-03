#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.run_incumbent_proxy_classic_teacher_diagnostic_trace import (
    format_bool,
    format_float,
    markdown_table,
    model_spec_from_metadata,
    row_specs_from_bucket_rows,
)
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    load_json,
    write_json,
)
from ml.alphazero_lite.run_incumbent_proxy_teacher_interference_attribution import (
    CLASSIC_ROW_IDS,
    EXCLUDED_ROW_IDS,
    LOW_LR,
    PUCT_ROW_IDS,
    TRACE_PHASE2_EPOCHS,
    VariantSpec,
    attribution_row_for_trace,
    choose_device,
    clone_artifact_row,
    current_checkpoint_path,
    evaluate_row_set_baseline,
    load_or_build_base_artifact,
    read_jsonl,
    removed_row_id_from_leave_one_out_variant,
    run_variant_trace,
)


DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_CLASSIC_ROWS_PATH = Path(
    "/tmp/azlite_incumbent_proxy_teacher_bucket_split/classic_teacher_rows.jsonl"
)
DEFAULT_BUCKET_REPORT_PATH = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-teacher-bucket-split-results.md"
)
DEFAULT_PUCT_ROWS_PATH = Path(
    "/tmp/azlite_incumbent_proxy_teacher_bucket_split/puct_teacher_rows.jsonl"
)
DEFAULT_EXCLUDED_ROWS_PATH = Path(
    "/tmp/azlite_incumbent_proxy_teacher_bucket_split/excluded_diagnostic_rows.jsonl"
)
DEFAULT_PR48_ARTIFACT_PATH = Path(
    "/tmp/azlite_incumbent_proxy_classic_teacher_diagnostic/classic_teacher_diagnostic_artifact.jsonl"
)
DEFAULT_ATTRIBUTION_SUMMARY_PATH = Path(
    "/tmp/azlite_incumbent_proxy_teacher_interference_attribution/interference_attribution_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/azlite_incumbent_proxy_classic_teacher_low_lr_non_regression_lane"
)
DEFAULT_ARTIFACT_PATH = DEFAULT_OUTPUT_DIR / "classic_all_low_lr_lane.jsonl"
DEFAULT_SUMMARY_PATH = (
    DEFAULT_OUTPUT_DIR / "classic_teacher_low_lr_non_regression_lane_summary.json"
)
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-incumbent-proxy-classic-teacher-low-lr-non-regression-lane-results.md"
)

SCHEMA = "azlite_incumbent_proxy_classic_teacher_low_lr_non_regression_lane_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument(
        "--classic-rows-path", type=Path, default=DEFAULT_CLASSIC_ROWS_PATH
    )
    parser.add_argument(
        "--bucket-report-path", type=Path, default=DEFAULT_BUCKET_REPORT_PATH
    )
    parser.add_argument("--puct-rows-path", type=Path, default=DEFAULT_PUCT_ROWS_PATH)
    parser.add_argument(
        "--excluded-rows-path", type=Path, default=DEFAULT_EXCLUDED_ROWS_PATH
    )
    parser.add_argument(
        "--pr48-artifact-path", type=Path, default=DEFAULT_PR48_ARTIFACT_PATH
    )
    parser.add_argument(
        "--attribution-summary-path",
        type=Path,
        default=DEFAULT_ATTRIBUTION_SUMMARY_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def materialize_lane_artifact(
    *, base_artifact_rows: list[dict[str, Any]], artifact_path: Path
) -> list[dict[str, Any]]:
    rows = [clone_artifact_row(row) for row in base_artifact_rows]
    write_jsonl(artifact_path, rows)
    return rows


def load_attribution_context(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "selected_best_leave_one_out": None,
            "recommendation_matches": False,
            "classification": None,
        }
    payload = load_json(path)
    classification = payload.get("decision", {}).get("classification")
    recommendation_matches = classification == "update_size_sensitive_interference"
    best_loo = None
    for row in list(payload.get("artifact_variants") or []):
        if row.get("artifact_name") == "best_leave_one_out_low_lr":
            excluded = list(row.get("excluded_classic_rows") or [])
            if len(excluded) == 1:
                excluded_row = str(excluded[0])
                for candidate_name in (
                    "classic_without_008",
                    "classic_without_014",
                    "classic_without_022",
                    "classic_without_024",
                    "classic_without_025",
                    "classic_without_035",
                ):
                    if (
                        removed_row_id_from_leave_one_out_variant(candidate_name)
                        == excluded_row
                    ):
                        best_loo = candidate_name
                        break
    return {
        "status": "loaded",
        "selected_best_leave_one_out": best_loo,
        "recommendation_matches": recommendation_matches,
        "classification": classification,
    }


def strict_gate(trace: dict[str, Any], epoch: int) -> dict[str, Any]:
    attribution = attribution_row_for_trace(trace, epoch)
    classic_rows = [row for row in trace["classic_rows"] if int(row["epoch"]) == epoch]
    puct_rows = [row for row in trace["puct_rows"] if int(row["epoch"]) == epoch]
    excluded_rows = [
        row for row in trace["excluded_rows"] if int(row["epoch"]) == epoch
    ]
    gate_failures: list[str] = []
    if attribution["classic_gain_count"] < 1:
        gate_failures.append("classic_gain_missing")
    if attribution["puct_lost_selection_384_count"] > 0:
        gate_failures.append("puct_lost_selection_384")
    if attribution["puct_lost_selection_1200_count"] > 0:
        gate_failures.append("puct_lost_selection_1200")
    if attribution["heavy_regression_count"] > 0:
        gate_failures.append("heavy_puct_regression")
    if attribution["excluded_drift_count"] > 0:
        gate_failures.append("excluded_drift")
    strict_classic_passes = sum(1 for row in classic_rows if bool(row["strict_pass"]))
    puct_full_match = sum(
        1
        for row in puct_rows
        if bool(row["selected_equals_puct_preferred_384"])
        and bool(row["selected_equals_puct_preferred_1200"])
    )
    return {
        "epoch": epoch,
        "classic_gain_count": attribution["classic_gain_count"],
        "classic_strict_pass_count": strict_classic_passes,
        "puct_full_match_count": puct_full_match,
        "puct_row_count": len(puct_rows),
        "excluded_drift_count": attribution["excluded_drift_count"],
        "heavy_regression_count": attribution["heavy_regression_count"],
        "gate_pass": not gate_failures,
        "gate_failures": gate_failures,
        "notes": "ok" if not gate_failures else ",".join(gate_failures),
        "excluded_rows_examined": len(excluded_rows),
    }


def classify_lane(gate_rows: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    passing_rows = [row for row in gate_rows if bool(row["gate_pass"])]
    if passing_rows:
        best = min(passing_rows, key=lambda row: int(row["epoch"]))
        return (
            "low_lr_lane_gate_passed",
            "keep this lane diagnostic-only, then prepare one controlled pre-arena candidate from the same low-LR configuration with the same strict PUCT gate.",
            [f"epoch {best['epoch']} cleared the strict PUCT non-regression gate"],
        )

    least_bad = min(
        gate_rows,
        key=lambda row: (
            row["heavy_regression_count"],
            len(row["gate_failures"]),
            -row["classic_gain_count"],
            row["epoch"],
        ),
    )
    return (
        "low_lr_lane_gate_failed",
        "do not advance this lane; keep the branch diagnostic-only because even low LR still fails the strict PUCT non-regression gate.",
        [
            f"best low-LR checkpoint was epoch {least_bad['epoch']} but still failed: {least_bad['notes']}",
        ],
    )


def build_report(summary: dict[str, Any]) -> str:
    trace = summary["trace"]
    gate_rows = summary["gate_rows"]
    classic_rows = [row for row in trace["classic_rows"] if int(row["epoch"]) > 0]
    puct_rows = [row for row in trace["puct_rows"] if int(row["epoch"]) > 0]

    lines = [
        "# AlphaZero-lite Incumbent Proxy Classic Teacher Low-LR Non-Regression Lane Results",
        "",
        "## 1. Context",
        "",
        "- This run implements the exact next branch recommended by the interference attribution audit.",
        "- It trains one low-LR Classic-only lane from the current checkpoint.",
        "- No arena was run.",
        "- No model was promoted.",
        "- No active references were mutated.",
        "- No PUCT or excluded rows were used as training targets.",
        "",
        "## 2. Lane Definition",
        "",
        f"- Artifact: `{summary['artifact_path']}`.",
        f"- LR: `{summary['trace']['lr']}`.",
        f"- Epoch checkpoints: `{', '.join(str(epoch) for epoch in summary['trace']['epochs'])}`.",
        f"- Attribution prerequisite status: `{summary['attribution_context']['status']}` with classification `{summary['attribution_context']['classification']}`.",
        "",
        "## 3. Strict Gate",
        "",
        "- Gate requires at least one Classic target gain, zero PUCT lost selections at 384 and 1200, zero heavy PUCT regressions, and zero excluded drift.",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "epoch",
                "classic_gain_count",
                "classic_strict_pass_count",
                "puct_full_match_count",
                "puct_row_count",
                "excluded_drift_count",
                "heavy_regression_count",
                "gate_pass",
                "notes",
            ],
            [
                [
                    str(row["epoch"]),
                    str(row["classic_gain_count"]),
                    str(row["classic_strict_pass_count"]),
                    str(row["puct_full_match_count"]),
                    str(row["puct_row_count"]),
                    str(row["excluded_drift_count"]),
                    str(row["heavy_regression_count"]),
                    format_bool(row["gate_pass"]),
                    row["notes"],
                ]
                for row in gate_rows
            ],
        )
    )
    lines.extend(["", "## 4. Classic Results", ""])
    lines.extend(
        markdown_table(
            [
                "epoch",
                "row_id",
                "selected_384",
                "selected_1200",
                "active_reference_move",
                "reference_visit_share_384",
                "reference_visit_share_1200",
                "improved_vs_current",
                "strict_pass",
                "notes",
            ],
            [
                [
                    str(row["epoch"]),
                    row["row_id"],
                    str(row["selected_move_384"]),
                    str(row["selected_move_1200"]),
                    str(row["active_reference_move"]),
                    format_float(row["reference_visit_share_384"]),
                    format_float(row["reference_visit_share_1200"]),
                    format_bool(row["improved_vs_current"]),
                    format_bool(row["strict_pass"]),
                    row["notes"],
                ]
                for row in classic_rows
            ],
        )
    )
    lines.extend(["", "## 5. PUCT Results", ""])
    lines.extend(
        markdown_table(
            [
                "epoch",
                "row_id",
                "puct_preferred_move",
                "selected_384",
                "selected_1200",
                "puct_preferred_visit_share_384",
                "puct_preferred_visit_share_1200",
                "selected_equals_puct_preferred_1200",
                "cross_teacher_regression",
                "notes",
            ],
            [
                [
                    str(row["epoch"]),
                    row["row_id"],
                    str(row["puct_preferred_move"]),
                    str(row["selected_move_384"]),
                    str(row["selected_move_1200"]),
                    format_float(row["puct_preferred_visit_share_384"]),
                    format_float(row["puct_preferred_visit_share_1200"]),
                    format_bool(row["selected_equals_puct_preferred_1200"]),
                    format_bool(row["cross_teacher_regression"]),
                    row["notes"],
                ]
                for row in puct_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 6. Decision",
            "",
            f"- Final classification: `{summary['classification']}`.",
        ]
    )
    for note in summary["classification_notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## 7. Exactly One Recommended Next Action", ""])
    lines.append(f"Recommendation: **{summary['recommended_next_action']}**")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attribution_context = load_attribution_context(args.attribution_summary_path)
    spec = VariantSpec(
        artifact_name="classic_all_low_lr_lane",
        included_row_ids=CLASSIC_ROW_IDS,
        notes="single low-LR follow-up lane from the interference attribution recommendation",
        epochs=TRACE_PHASE2_EPOCHS,
        lr=LOW_LR,
    )
    lane_artifact_path = spec.artifact_path
    base_artifact_rows = load_or_build_base_artifact(args)
    materialize_lane_artifact(
        base_artifact_rows=base_artifact_rows,
        artifact_path=lane_artifact_path,
    )

    current_metadata = load_json(args.current_artifact / "metadata.json")
    base_model_spec = model_spec_from_metadata(current_metadata)
    init_checkpoint = current_checkpoint_path(args.current_artifact, args.output_dir)
    device = choose_device(args.device)

    classic_rows = read_jsonl(args.classic_rows_path)
    puct_rows = read_jsonl(args.puct_rows_path)
    excluded_rows = read_jsonl(args.excluded_rows_path)
    classic_specs = row_specs_from_bucket_rows(classic_rows, CLASSIC_ROW_IDS)
    puct_specs = row_specs_from_bucket_rows(puct_rows, PUCT_ROW_IDS)
    excluded_specs = row_specs_from_bucket_rows(excluded_rows, EXCLUDED_ROW_IDS)
    classic_spec_by_id = {row["row_id"]: row for row in classic_specs}
    puct_spec_by_id = {row["row_id"]: row for row in puct_specs}
    excluded_spec_by_id = {row["row_id"]: row for row in excluded_specs}

    classic_baseline_by_id = {
        row["row_id"]: row
        for row in evaluate_row_set_baseline(
            artifact_path=args.current_artifact,
            row_specs=classic_specs,
            kind="classic",
            seed=args.seed,
        )
    }
    puct_baseline_by_id = {
        row["row_id"]: row
        for row in evaluate_row_set_baseline(
            artifact_path=args.current_artifact,
            row_specs=puct_specs,
            kind="puct",
            seed=args.seed + 1000,
        )
    }
    excluded_baseline_by_id = {
        row["row_id"]: row
        for row in evaluate_row_set_baseline(
            artifact_path=args.current_artifact,
            row_specs=excluded_specs,
            kind="excluded",
            seed=args.seed + 2000,
        )
    }

    trace = run_variant_trace(
        spec=spec,
        classic_spec_by_id=classic_spec_by_id,
        puct_spec_by_id=puct_spec_by_id,
        excluded_spec_by_id=excluded_spec_by_id,
        classic_baseline_by_id=classic_baseline_by_id,
        puct_baseline_by_id=puct_baseline_by_id,
        excluded_baseline_by_id=excluded_baseline_by_id,
        init_checkpoint=init_checkpoint,
        current_metadata=current_metadata,
        base_model_spec=base_model_spec,
        export_root=args.output_dir / "exports",
        seed=args.seed,
        device=device,
    )
    trace["classic_baseline_by_id"] = classic_baseline_by_id
    trace["puct_baseline_by_id"] = puct_baseline_by_id
    trace["excluded_baseline_by_id"] = excluded_baseline_by_id
    trace["artifact_path"] = str(lane_artifact_path)

    gate_rows = [strict_gate(trace, epoch) for epoch in TRACE_PHASE2_EPOCHS]
    classification, recommended_next_action, classification_notes = classify_lane(
        gate_rows
    )

    summary = {
        "schema": SCHEMA,
        "artifact_path": str(lane_artifact_path),
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "classic_rows_path": str(args.classic_rows_path),
            "bucket_report_path": str(args.bucket_report_path),
            "puct_rows_path": str(args.puct_rows_path),
            "excluded_rows_path": str(args.excluded_rows_path),
            "pr48_artifact_path": str(args.pr48_artifact_path),
            "attribution_summary_path": str(args.attribution_summary_path),
            "init_checkpoint": str(init_checkpoint),
        },
        "attribution_context": attribution_context,
        "classification": classification,
        "classification_notes": classification_notes,
        "recommended_next_action": recommended_next_action,
        "acceptance": {
            "arena_run": False,
            "mcts1200_lane_run": False,
            "model_promoted": False,
            "active_references_mutated": False,
            "puct_rows_used_as_targets": False,
            "excluded_rows_used_as_targets": False,
        },
        "gate_rows": gate_rows,
        "trace": trace,
    }
    write_json(args.summary_out, summary)
    args.report_path.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.summary_out),
                "report_path": str(args.report_path),
                "classification": classification,
                "recommended_next_action": recommended_next_action,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
