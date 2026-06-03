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
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/azlite_incumbent_proxy_classic_teacher_soft_policy_non_regression_lane"
)
DEFAULT_ARTIFACT_DIR = DEFAULT_OUTPUT_DIR / "artifacts"
DEFAULT_SUMMARY_PATH = (
    DEFAULT_OUTPUT_DIR / "classic_teacher_soft_policy_non_regression_lane_summary.json"
)
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-incumbent-proxy-classic-teacher-soft-policy-non-regression-lane-results.md"
)

SCHEMA = "azlite_incumbent_proxy_classic_teacher_soft_policy_non_regression_lane_v1"
POLICY_REFERENCE_MASSES = (0.7, 0.6)


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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
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


def soft_policy(
    legal_moves: list[int], reference_move: int, reference_mass: float
) -> list[float]:
    policy = [0.0] * 6
    if reference_move not in legal_moves:
        raise ValueError(f"reference move {reference_move} illegal for softened policy")
    if len(legal_moves) == 1:
        policy[reference_move] = 1.0
        return policy
    non_reference_mass = 1.0 - float(reference_mass)
    residual = non_reference_mass / float(len(legal_moves) - 1)
    for move in legal_moves:
        policy[int(move)] = residual
    policy[int(reference_move)] = float(reference_mass)
    return policy


def materialize_soft_policy_artifact(
    *,
    base_artifact_rows: list[dict[str, Any]],
    artifact_path: Path,
    reference_mass: float,
) -> None:
    rows: list[dict[str, Any]] = []
    for row in base_artifact_rows:
        cloned = clone_artifact_row(row)
        legal_moves = [int(move) for move in cloned["legal_moves"]]
        reference_move = int(cloned["active_reference_move"])
        cloned["policy"] = soft_policy(legal_moves, reference_move, reference_mass)
        cloned["policy_target_reference_mass"] = float(reference_mass)
        cloned["policy_target_non_reference_mass"] = round(
            1.0 - float(reference_mass), 6
        )
        cloned["selection_reasons"] = list(cloned.get("selection_reasons") or []) + [
            f"soft_policy_{reference_mass:.2f}"
        ]
        rows.append(cloned)
    write_jsonl(artifact_path, rows)


def strict_gate(trace: dict[str, Any], epoch: int) -> dict[str, Any]:
    attribution = attribution_row_for_trace(trace, epoch)
    classic_rows = [row for row in trace["classic_rows"] if int(row["epoch"]) == epoch]
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
    return {
        "artifact_name": trace["artifact_name"],
        "epoch": epoch,
        "classic_gain_count": attribution["classic_gain_count"],
        "classic_strict_pass_count": sum(
            1 for row in classic_rows if bool(row["strict_pass"])
        ),
        "puct_lost_selection_384_count": attribution["puct_lost_selection_384_count"],
        "puct_lost_selection_1200_count": attribution["puct_lost_selection_1200_count"],
        "heavy_regression_count": attribution["heavy_regression_count"],
        "excluded_drift_count": attribution["excluded_drift_count"],
        "gate_pass": not gate_failures,
        "notes": "ok" if not gate_failures else ",".join(gate_failures),
    }


def classify_results(gate_rows: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    passing_rows = [row for row in gate_rows if bool(row["gate_pass"])]
    if passing_rows:
        best = min(
            passing_rows,
            key=lambda row: (row["epoch"], row["artifact_name"]),
        )
        return (
            "soft_policy_gate_passed",
            "run one tiny softened-policy confirmation lane with the passing reference mass only, still no arena.",
            [
                f"{best['artifact_name']} passed the strict gate at epoch {best['epoch']}",
            ],
        )
    least_bad = min(
        gate_rows,
        key=lambda row: (
            row["heavy_regression_count"],
            row["puct_lost_selection_1200_count"],
            row["excluded_drift_count"],
            -row["classic_gain_count"],
            row["artifact_name"],
            row["epoch"],
        ),
    )
    return (
        "soft_policy_gate_pending",
        "use this prepared softened-policy branch for the next diagnostic run; do not advance any replay lane until one softened-policy variant clears the strict PUCT gate.",
        [
            f"best prepared candidate so far is {least_bad['artifact_name']} at epoch {least_bad['epoch']} by the gate heuristic",
        ],
    )


def build_report(summary: dict[str, Any]) -> str:
    gate_rows = summary["gate_rows"]
    lines = [
        "# AlphaZero-lite Incumbent Proxy Classic Teacher Soft-Policy Non-Regression Lane Results",
        "",
        "## 1. Context",
        "",
        "- This branch is prepared because the dedicated low-LR lane still failed the strict PUCT non-regression gate.",
        "- It softens Classic policy targets while keeping the same current-checkpoint initializer and low learning rate.",
        "- No arena is part of this branch.",
        "",
        "## 2. Prepared Variants",
        "",
    ]
    lines.extend(
        markdown_table(
            ["artifact_name", "reference_mass", "epochs", "lr", "status"],
            [
                [
                    row["artifact_name"],
                    str(row["reference_mass"]),
                    ", ".join(str(epoch) for epoch in row["epochs"]),
                    str(row["lr"]),
                    row["status"],
                ]
                for row in summary["prepared_variants"]
            ],
        )
    )
    lines.extend(["", "## 3. Strict Gate Template", ""])
    lines.extend(
        markdown_table(
            [
                "artifact_name",
                "epoch",
                "classic_gain_count",
                "puct_lost_selection_384_count",
                "puct_lost_selection_1200_count",
                "heavy_regression_count",
                "excluded_drift_count",
                "gate_pass",
                "notes",
            ],
            [
                [
                    row["artifact_name"],
                    str(row["epoch"]),
                    str(row["classic_gain_count"]),
                    str(row["puct_lost_selection_384_count"]),
                    str(row["puct_lost_selection_1200_count"]),
                    str(row["heavy_regression_count"]),
                    str(row["excluded_drift_count"]),
                    format_bool(row["gate_pass"]),
                    row["notes"],
                ]
                for row in gate_rows
            ],
        )
    )
    lines.extend(["", "## 4. Decision", ""])
    lines.append(f"- Classification: `{summary['classification']}`.")
    for note in summary["classification_notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## 5. Exactly One Recommended Next Action", ""])
    lines.append(f"Recommendation: **{summary['recommended_next_action']}**")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    base_artifact_rows = load_or_build_base_artifact(args)
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

    prepared_variants: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for reference_mass in POLICY_REFERENCE_MASSES:
        artifact_name = (
            f"classic_all_soft_policy_r{int(reference_mass * 100):02d}_low_lr"
        )
        spec = VariantSpec(
            artifact_name=artifact_name,
            included_row_ids=CLASSIC_ROW_IDS,
            notes=f"softened Classic policy targets with reference mass {reference_mass:.2f}",
            epochs=TRACE_PHASE2_EPOCHS,
            lr=LOW_LR,
        )
        materialize_soft_policy_artifact(
            base_artifact_rows=base_artifact_rows,
            artifact_path=spec.artifact_path,
            reference_mass=reference_mass,
        )
        prepared_variants.append(
            {
                "artifact_name": artifact_name,
                "artifact_path": str(spec.artifact_path),
                "reference_mass": reference_mass,
                "epochs": list(spec.epochs),
                "lr": LOW_LR,
                "status": "prepared",
            }
        )
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
        traces.append(trace)
        for epoch in TRACE_PHASE2_EPOCHS:
            gate_rows.append(strict_gate(trace, epoch))

    classification, recommended_next_action, classification_notes = classify_results(
        gate_rows
    )
    summary = {
        "schema": SCHEMA,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "classic_rows_path": str(args.classic_rows_path),
            "bucket_report_path": str(args.bucket_report_path),
            "puct_rows_path": str(args.puct_rows_path),
            "excluded_rows_path": str(args.excluded_rows_path),
            "pr48_artifact_path": str(args.pr48_artifact_path),
            "init_checkpoint": str(init_checkpoint),
        },
        "prepared_variants": prepared_variants,
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
        "traces": traces,
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
