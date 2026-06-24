#!/usr/bin/env python3
"""Balanced-current PUCT disagreement/stability iteration smoke test."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.run_balanced_opening_puct_replay import (  # noqa: E402
    PRIMARY_BUDGET,
    EQ_768_BUDGET,
    EQ_1200_BUDGET,
    ASYM_1200_256_BUDGET,
    aggregate_budget_summary,
    anchored_top1_preserved_rate,
    build_stability_replay,
    collect_stability_candidates,
    effective_sampling_fractions,
    large_suite_rows,
    load_candidate_policy_shift_rows,
    markdown_table,
    parse_budget_pairs,
    parse_csv_paths,
    select_stability_rows,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    benchmark_budget_results,
    bootstrap_ci,
    gate_budget_results,
    pooled_per_opening_differences,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    MIN_TRAINABLE_ROWS,
    analyze_state_batch,
    build_input_summary,
    collect_evaluation_states,
    continue_state_batch,
    mined_state_policy_shift,
    partition_batches,
    percentile,
    policy_entropy,
    read_jsonl,
    require_existing_file,
    select_replay_rows,
    sha256_file,
    summarize_disagreement_audit,
    verify_expected_hash,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.run_promoted_current_puct_iter2_smoke import (  # noqa: E402
    compute_param_delta_norm,
    export_checkpoint,
    find_candidate_report,
    heldout_summary,
    run_default_gate,
    run_opening_suite_benchmark,
    run_train,
)

SUMMARY_SCHEMA = "azlite_balanced_current_iter2_smoke_v1"
REPORT_PATH = REPO_ROOT / "docs/alphazero-lite-balanced-current-iter2-smoke-results.md"
DISAGREEMENT_REPLAY_FILENAME = "current_opening_disagreement_replay.jsonl"
STABILITY_REPLAY_FILENAME = "current_equal_budget_stability_replay.jsonl"
PUCT_AUDIT_FILENAME = "current_puct_audit.json"
SUMMARY_FILENAME = "summary_metrics.json"
DISAGREEMENT_TRACE_FILENAME = "current_disagreement_states.jsonl"
STABILITY_TRACE_FILENAME = "current_stability_candidates.jsonl"
STABILITY_SELECTED_FILENAME = "current_stability_selected.jsonl"
TARGET_ROWS = 2000
GATE_DELTA_THRESHOLD = 0.05
PR126_AUDIT_PATH = Path(
    "/tmp/azlite_promoted_current_opening_puct_disagreement/opening_state_disagreement_audit.json"
)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/tmp/azlite_balanced_current_iter2")
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--expected-init-checkpoint-sha256", required=True)
    parser.add_argument("--generic-bootstrap", required=True)
    parser.add_argument("--random-teacher", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--disagreement-weight", type=int, default=8)
    parser.add_argument("--stability-weight", type=int, default=4)
    parser.add_argument(
        "--budget-pairs",
        default="384:256,768:256,768:768,1200:1200,1200:256,256:768",
    )
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--simulations", type=int, default=384)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def validate_replay_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_legal_mask_rows = 0
    invalid_value_rows = 0
    for row in rows:
        legal_moves = {int(move) for move in row.get("legal_moves", [])}
        policy = list(row.get("policy", []))
        total = sum(float(value) for value in policy)
        if len(policy) != 6 or abs(total - 1.0) > 1e-6:
            invalid_legal_mask_rows += 1
        else:
            for move, probability in enumerate(policy):
                if move not in legal_moves and float(probability) > 1e-6:
                    invalid_legal_mask_rows += 1
                    break
        value = float(row.get("value", 0.0))
        if value < -1.0 or value > 1.0:
            invalid_value_rows += 1
    return {
        "legal_mask_valid": invalid_legal_mask_rows == 0,
        "invalid_legal_mask_rows": invalid_legal_mask_rows,
        "value_target_valid": invalid_value_rows == 0,
        "invalid_value_target_rows": invalid_value_rows,
    }


def summarize_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_entropies = [
        float(policy_entropy(row["raw_policy"], row["legal_moves"])) for row in rows
    ]
    search_entropies = [
        float(row.get("search_entropy", 0.0))
        for row in rows
        if row.get("search_entropy") is not None
    ]
    visit_shares = [float(row["search_top1_visit_share"]) for row in rows]
    phase_counts = Counter(str(row["phase"]) for row in rows)
    margin_buckets = Counter()
    top_move_counts = Counter()
    for row in rows:
        margin = float(row.get("raw_margin", 0.0))
        if margin < 0.02:
            margin_buckets["margin < 0.02"] += 1
        elif margin < 0.05:
            margin_buckets["0.02 <= margin < 0.05"] += 1
        elif margin < 0.10:
            margin_buckets["0.05 <= margin < 0.10"] += 1
        else:
            margin_buckets["margin >= 0.10"] += 1
        if row.get("search_top1") is not None:
            top_move_counts[str(int(row["search_top1"]))] += 1
    return {
        "policy_entropy": {
            "raw_mean": statistics.fmean(raw_entropies) if raw_entropies else 0.0,
            "raw_p50": percentile(raw_entropies, 50),
            "raw_p90": percentile(raw_entropies, 90),
            "search_mean": statistics.fmean(search_entropies)
            if search_entropies
            else 0.0,
            "search_p50": percentile(search_entropies, 50),
            "search_p90": percentile(search_entropies, 90),
        },
        "search_top1_visit_share": {
            "mean": statistics.fmean(visit_shares) if visit_shares else 0.0,
            "p50": percentile(visit_shares, 50),
            "p90": percentile(visit_shares, 90),
        },
        "raw_top1_margin_buckets": dict(sorted(margin_buckets.items())),
        "top_move_distribution_by_pit": dict(sorted(top_move_counts.items())),
        "phase_distribution": dict(sorted(phase_counts.items())),
    }


def disagreement_audit_summary(
    analyzed_states: list[dict[str, Any]],
    collection_metadata: dict[str, Any],
    replay_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit = summarize_disagreement_audit(analyzed_states, collection_metadata)
    selected_states, selection_summary = select_replay_rows(analyzed_states)
    audit["tables"]["selection_summary"] = selection_summary
    audit["tables"]["trainable_disagreement_rows_after_dedupe"] = len(selected_states)
    audit["tables"]["replay_validation"] = validate_replay_rows(replay_rows)
    audit["tables"]["distribution"] = summarize_distribution(analyzed_states)
    return audit, selected_states


def build_pr126_comparison(current_audit: dict[str, Any]) -> dict[str, Any]:
    if not PR126_AUDIT_PATH.is_file():
        return {"available": False}
    pr126_audit = load_json(PR126_AUDIT_PATH)
    current_tables = current_audit.get("tables", {})
    prior_tables = pr126_audit.get("tables", {})
    current_rate = float(
        current_tables.get("overall_disagreement_rate", {}).get("rate", 0.0)
    )
    prior_rate = float(
        prior_tables.get("overall_disagreement_rate", {}).get("rate", 0.0)
    )
    current_hc = int(current_tables.get("high_confidence_disagreements", 0))
    prior_hc = int(prior_tables.get("high_confidence_disagreements", 0))
    current_kl = float(
        current_tables.get("top_raw_margin_buckets", {})
        .get("margin >= 0.10", {})
        .get("mean_kl_search_raw", 0.0)
    )
    prior_kl = float(
        prior_tables.get("top_raw_margin_buckets", {})
        .get("margin >= 0.10", {})
        .get("mean_kl_search_raw", 0.0)
    )
    return {
        "available": True,
        "path": str(PR126_AUDIT_PATH),
        "disagreement_rate": {
            "current": current_rate,
            "pr126": prior_rate,
            "delta": current_rate - prior_rate,
        },
        "high_confidence_disagreements": {
            "current": current_hc,
            "pr126": prior_hc,
            "delta": current_hc - prior_hc,
        },
        "mean_kl_search_raw_margin_ge_0_10": {
            "current": current_kl,
            "pr126": prior_kl,
            "delta": current_kl - prior_kl,
        },
    }


def build_puct_audit(
    *,
    disagreement_analyzed_states: list[dict[str, Any]],
    disagreement_replay_rows: list[dict[str, Any]],
    disagreement_audit: dict[str, Any],
    stability_candidate_rows: list[dict[str, Any]],
    stability_selected_rows: list[dict[str, Any]],
    stability_replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    disagreement_hashes = {str(row["state_hash"]) for row in disagreement_replay_rows}
    stability_hashes = {str(row["state_hash"]) for row in stability_replay_rows}
    overlap = sorted(disagreement_hashes & stability_hashes)
    stability_validation = validate_replay_rows(stability_replay_rows)
    return {
        "schema": "azlite_balanced_current_iter2_puct_audit_v1",
        "disagreement": {
            "analyzed_state_count": len(disagreement_analyzed_states),
            "top1_rate": disagreement_audit["tables"]["overall_disagreement_rate"],
            "high_confidence_disagreement_count": disagreement_audit["tables"][
                "high_confidence_disagreements"
            ],
            "row_count": len(disagreement_replay_rows),
            "unique_row_count": len(disagreement_hashes),
            "distribution": disagreement_audit["tables"]["distribution"],
            "legal_mask_validity": disagreement_audit["tables"]["replay_validation"],
            "kl": {
                "search_to_raw_mean": statistics.fmean(
                    float(row["kl_search_raw"]) for row in disagreement_analyzed_states
                )
                if disagreement_analyzed_states
                else 0.0,
                "raw_to_search_mean": statistics.fmean(
                    float(row["kl_raw_search"]) for row in disagreement_analyzed_states
                )
                if disagreement_analyzed_states
                else 0.0,
            },
        },
        "stability": {
            "candidate_count": len(stability_candidate_rows),
            "row_count": len(stability_replay_rows),
            "unique_row_count": len(stability_hashes),
            "selection_summary": {
                "selected_rows": len(stability_selected_rows),
                "unique_rows": len(
                    {str(row["state_hash"]) for row in stability_selected_rows}
                ),
            },
            "distribution": summarize_distribution(stability_selected_rows),
            "legal_mask_validity": stability_validation,
        },
        "replay_overlap": {
            "count": len(overlap),
            "valid": not overlap,
        },
    }


def gate_targets_from_summary(candidate_rows: list[dict[str, Any]]) -> list[str]:
    targets = ["balanced_current_ref"]
    current_row = next(
        row for row in candidate_rows if row["candidate"] == "balanced_current_ref"
    )
    current_mean = float(
        current_row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]
    )
    for row in candidate_rows:
        if row["candidate"] == "balanced_current_ref":
            continue
        mean_ds = float(row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"])
        if mean_ds >= current_mean + GATE_DELTA_THRESHOLD:
            targets.append(str(row["candidate"]))
    return targets


def build_candidate_summary_row(
    *,
    candidate: dict[str, Any],
    init_checkpoint: Path,
    medium_report: dict[str, Any],
    fixed_large_report: dict[str, Any],
    heldout_reports: dict[str, dict[str, Any]],
    suite_rows: dict[str, Any],
    budget_pairs: list[str],
    disagreement_policy_rows: list[dict[str, Any]],
    stability_policy_rows: list[dict[str, Any]],
    seed: int,
    gate_report: dict[str, Any] | None,
) -> dict[str, Any]:
    checkpoint_path = Path(str(candidate["checkpoint_path"]))
    artifact_dir = Path(str(candidate["artifact_dir"]))
    delta_norm, relative_delta_pct = compute_param_delta_norm(
        checkpoint_path, init_checkpoint
    )
    row: dict[str, Any] = {
        "candidate": candidate["name"],
        "report_candidate_name": candidate["report_candidate_name"],
        "epochs": candidate["epochs"],
        "trainable_scope": candidate["trainable_scope"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "artifact_dir": str(artifact_dir),
        "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
        "delta_norm_vs_current_checkpoint": delta_norm,
        "relative_delta_pct_vs_current_checkpoint": relative_delta_pct,
        "medium_budget_results": benchmark_budget_results(
            find_candidate_report(
                medium_report, str(candidate["report_candidate_name"])
            )
        ),
        "fixed_large_budget_results": benchmark_budget_results(
            find_candidate_report(
                fixed_large_report, str(candidate["report_candidate_name"])
            )
        ),
        "heldout_summary": heldout_summary(
            heldout_reports, str(candidate["report_candidate_name"])
        ),
        "large_suite_aggregate": {
            budget_pair: aggregate_budget_summary(
                suite_rows,
                str(candidate["name"]),
                budget_pair,
                "balanced_current_ref",
            )
            for budget_pair in budget_pairs
        },
        "mined_state_policy_shift": mined_state_policy_shift(
            artifact_dir, disagreement_policy_rows
        ),
        "stability_anchor_top1_preserved_rate": anchored_top1_preserved_rate(
            artifact_dir, stability_policy_rows
        ),
    }
    if candidate.get("train_metrics"):
        row["policy_loss"] = candidate["train_metrics"].get("policy_loss")
        row["value_loss"] = candidate["train_metrics"].get("value_loss")
        row["validation_loss"] = candidate["train_metrics"].get("best_val_loss")
        row["training_elapsed_s"] = candidate["train_metrics"].get("training_elapsed_s")
    if candidate.get("effective_replay_sampling_fractions"):
        row["effective_replay_sampling_fractions"] = candidate[
            "effective_replay_sampling_fractions"
        ]
    row["bootstrap_cis"] = {}
    for budget_pair in (
        PRIMARY_BUDGET,
        EQ_768_BUDGET,
        EQ_1200_BUDGET,
        ASYM_1200_256_BUDGET,
    ):
        diffs = pooled_per_opening_differences(
            suite_rows=suite_rows,
            candidate_a=str(candidate["name"]),
            candidate_b="balanced_current_ref",
            budget_pair=budget_pair,
            metric_key="ds",
        )
        row["bootstrap_cis"][
            f"{candidate['name']}_minus_balanced_current_ref_{budget_pair.replace(':', '_')}"
        ] = bootstrap_ci(diffs, seed=seed, samples=DEFAULT_BOOTSTRAP_SAMPLES)
    if gate_report is not None:
        row["default_gate"] = {
            "classification": gate_report.get("classification"),
            "budget_results": gate_budget_results(gate_report),
        }
        row["high_search_breakthrough_preserved"] = (
            gate_report.get("classification") == "high_search_breakthrough"
        )
    else:
        row["high_search_breakthrough_preserved"] = None
    return row


def classify_run(
    summary_candidates: list[dict[str, Any]], puct_audit: dict[str, Any]
) -> str:
    disagreement_rows = int(puct_audit["disagreement"]["unique_row_count"])
    high_confidence_count = int(
        puct_audit["disagreement"]["high_confidence_disagreement_count"]
    )
    analyzed_states = max(int(puct_audit["disagreement"]["analyzed_state_count"]), 1)
    high_confidence_rate = high_confidence_count / analyzed_states
    if disagreement_rows < MIN_TRAINABLE_ROWS or high_confidence_rate < 0.05:
        return "current_search_saturated"

    current_row = next(
        row for row in summary_candidates if row["candidate"] == "balanced_current_ref"
    )
    iter_rows = [
        row
        for row in summary_candidates
        if row["candidate"].startswith("balanced_iter2_")
    ]
    best = max(
        iter_rows,
        key=lambda row: float(row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]),
    )
    delta_384 = float(best["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]) - float(
        current_row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]
    )
    delta_768 = float(best["large_suite_aggregate"][EQ_768_BUDGET]["mean_ds"]) - float(
        current_row["large_suite_aggregate"][EQ_768_BUDGET]["mean_ds"]
    )
    delta_1200 = float(
        best["large_suite_aggregate"][EQ_1200_BUDGET]["mean_ds"]
    ) - float(current_row["large_suite_aggregate"][EQ_1200_BUDGET]["mean_ds"])
    delta_1200_256 = float(
        best["large_suite_aggregate"][ASYM_1200_256_BUDGET]["mean_ds"]
    ) - float(current_row["large_suite_aggregate"][ASYM_1200_256_BUDGET]["mean_ds"])
    ci_384 = best["bootstrap_cis"][
        f"{best['candidate']}_minus_balanced_current_ref_384_256"
    ]
    heldout_current = current_row.get("heldout_summary", {})
    heldout_best = best.get("heldout_summary", {})
    heldout_delta = None
    if heldout_current.get("available") and heldout_best.get("available"):
        heldout_delta = float(heldout_best["mean_ds_384_256"]) - float(
            heldout_current["mean_ds_384_256"]
        )
    fixed_delta = delta_384
    p0_p1_gap_best = abs(
        float(best["large_suite_aggregate"][PRIMARY_BUDGET]["mean_p0_score"])
        - float(best["large_suite_aggregate"][PRIMARY_BUDGET]["mean_p1_score"])
    )
    p0_p1_gap_current = abs(
        float(current_row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_p0_score"])
        - float(current_row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_p1_score"])
    )
    gate_ok = best.get("high_search_breakthrough_preserved") is not False

    if (
        delta_384 >= 0.08
        and float(ci_384["lower"]) > 0.03
        and delta_768 >= -0.08
        and delta_1200 >= -0.03
        and delta_1200_256 >= -0.03
        and gate_ok
    ):
        return "balanced_iter2_improvement"
    if (
        (fixed_delta > 0.0 and heldout_delta is not None and heldout_delta <= 0.0)
        or delta_768 < -0.10
        or p0_p1_gap_best > p0_p1_gap_current + 0.10
    ):
        return "iter2_overfit_or_destructive"
    if delta_384 < 0.03 and delta_768 >= -0.08:
        return "stability_dominated"
    if (
        0.03 <= delta_384 < 0.08
        and delta_768 >= -0.08
        and heldout_delta is not None
        and heldout_delta >= 0.03
    ):
        return "balanced_iter2_borderline"
    return "iter2_overfit_or_destructive"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-Lite Balanced Current Iter2 Smoke Results",
        "",
        f"**Date**: {date.today().isoformat()}",
        "",
        f"**Classification**: `{summary['classification']}`",
        "",
        "## Inputs",
        "",
        f"- Current artifact weights SHA256: `{summary['inputs']['current_artifact_weights']['actual_sha256']}`",
        f"- Current source checkpoint SHA256: `{summary['inputs']['init_checkpoint']['actual_sha256']}`",
        "",
        "## Replay Audit",
        "",
        f"- Disagreement analyzed states: `{summary['puct_audit']['disagreement']['analyzed_state_count']}`",
        f"- Disagreement top-1 rate: `{summary['puct_audit']['disagreement']['top1_rate']['rate']:.4f}`",
        f"- High-confidence disagreements: `{summary['puct_audit']['disagreement']['high_confidence_disagreement_count']}`",
        f"- Disagreement replay rows: `{summary['puct_audit']['disagreement']['row_count']}`",
        f"- Stability candidate rows: `{summary['puct_audit']['stability']['candidate_count']}`",
        f"- Stability replay rows: `{summary['puct_audit']['stability']['row_count']}`",
        f"- Replay overlap: `{summary['puct_audit']['replay_overlap']['count']}`",
        "",
    ]
    if summary["puct_audit"].get("pr126_comparison", {}).get("available"):
        comp = summary["puct_audit"]["pr126_comparison"]
        lines.extend(
            [
                "## PR126 Comparison",
                "",
                f"- Disagreement-rate delta vs PR126 audit: `{fmt(float(comp['disagreement_rate']['delta']))}`",
                f"- High-confidence disagreement delta vs PR126 audit: `{comp['high_confidence_disagreements']['delta']}`",
                f"- High-margin KL delta vs PR126 audit: `{fmt(float(comp['mean_kl_search_raw_margin_ge_0_10']['delta']))}`",
                "",
            ]
        )
    rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        agg = candidate["large_suite_aggregate"]
        rows.append(
            [
                candidate["candidate"],
                fmt(float(agg[PRIMARY_BUDGET]["mean_ds"])),
                fmt(
                    float(
                        agg[PRIMARY_BUDGET]["mean_ds"]
                        - summary["current_large_mean_384_256"]
                    )
                ),
                fmt(
                    float(
                        agg[EQ_768_BUDGET]["mean_ds"]
                        - summary["current_large_mean_768_768"]
                    )
                ),
                fmt(
                    float(
                        agg[EQ_1200_BUDGET]["mean_ds"]
                        - summary["current_large_mean_1200_1200"]
                    )
                ),
                fmt(
                    float(
                        candidate["stability_anchor_top1_preserved_rate"][
                            "top1_preserved_rate"
                        ]
                    )
                ),
                fmt(
                    float(
                        candidate["mined_state_policy_shift"][
                            "top1_changed_rate_vs_promoted_current"
                        ]
                    )
                ),
            ]
        )
    lines.extend(
        [
            "## Candidate Aggregate Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Mean DS 384:256",
                    "Delta 384:256",
                    "Delta 768:768",
                    "Delta 1200:1200",
                    "Stability preserved",
                    "Mined top-1 changed",
                ],
                rows,
            ),
            "",
        ]
    )
    ci_rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        for key, ci in sorted(candidate.get("bootstrap_cis", {}).items()):
            ci_rows.append(
                [
                    key,
                    fmt(float(ci["mean"])),
                    fmt(float(ci["lower"])),
                    fmt(float(ci["upper"])),
                    ci["n"],
                ]
            )
    lines.extend(
        [
            "## Bootstrap CI Table",
            "",
            markdown_table(
                ["Comparison", "Mean", "Lower 95%", "Upper 95%", "Openings"], ci_rows
            ),
            "",
        ]
    )
    fixed_rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        budget_results = candidate["fixed_large_budget_results"]
        fixed_rows.append(
            [
                candidate["candidate"],
                fmt(float(budget_results[PRIMARY_BUDGET]["ds"])),
                fmt(float(budget_results["768:256"]["ds"])),
                fmt(float(budget_results[EQ_768_BUDGET]["ds"])),
                fmt(float(budget_results[EQ_1200_BUDGET]["ds"])),
                fmt(float(budget_results[ASYM_1200_256_BUDGET]["ds"])),
                fmt(float(budget_results["256:768"]["ds"])),
            ]
        )
    lines.extend(
        [
            "## Fixed Large DS Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "384:256",
                    "768:256",
                    "768:768",
                    "1200:1200",
                    "1200:256",
                    "256:768",
                ],
                fixed_rows,
            ),
            "",
        ]
    )
    heldout_rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        heldout = candidate["heldout_summary"]
        heldout_rows.append(
            [
                candidate["candidate"],
                fmt(float(heldout.get("mean_ds_384_256", 0.0)))
                if heldout.get("available")
                else "n/a",
                fmt(float(heldout.get("worst_suite_ds_384_256", 0.0)))
                if heldout.get("available")
                else "n/a",
            ]
        )
    lines.extend(
        [
            "## Held-Out Mean/Worst DS Table",
            "",
            markdown_table(
                ["Candidate", "Held-out mean 384:256", "Held-out worst-suite 384:256"],
                heldout_rows,
            ),
            "",
        ]
    )
    split_rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        agg = candidate["large_suite_aggregate"][PRIMARY_BUDGET]
        split_rows.append(
            [
                candidate["candidate"],
                fmt(float(agg["mean_p0_score"])),
                fmt(float(agg["mean_p1_score"])),
                fmt(abs(float(agg["mean_p0_score"]) - float(agg["mean_p1_score"]))),
                fmt(float(agg["mean_duplicate_trajectory_count"])),
            ]
        )
    lines.extend(
        [
            "## P0/P1 Split At 384:256",
            "",
            markdown_table(
                ["Candidate", "Mean P0", "Mean P1", "Gap", "Mean duplicates"],
                split_rows,
            ),
            "",
            "## Gate",
            "",
        ]
    )
    for candidate in summary["candidates"]:
        gate = candidate.get("default_gate")
        if gate is None:
            lines.append(f"- {candidate['candidate']}: `not_run`")
        else:
            lines.append(f"- {candidate['candidate']}: `{gate['classification']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_artifact = Path(args.current)
    init_checkpoint = Path(args.init_checkpoint)
    generic_bootstrap = Path(args.generic_bootstrap)
    random_teacher = Path(args.random_teacher)
    fixed_large_suite = Path(args.fixed_large_suite)
    medium_suite = Path(args.medium_suite)
    heldout_suite_paths = parse_csv_paths(args.heldout_suites)
    budget_pairs = parse_budget_pairs(args.budget_pairs)
    suite_specs = [("fixed_large", fixed_large_suite)] + [
        (path.stem, path) for path in heldout_suite_paths
    ]
    disagreement_trace_path = workdir / DISAGREEMENT_TRACE_FILENAME
    disagreement_replay_path = workdir / DISAGREEMENT_REPLAY_FILENAME
    stability_trace_path = workdir / STABILITY_TRACE_FILENAME
    stability_selected_path = workdir / STABILITY_SELECTED_FILENAME
    stability_replay_path = workdir / STABILITY_REPLAY_FILENAME

    for path, label in (
        (current_artifact / "weights.json", "current artifact weights"),
        (current_artifact / "metadata.json", "current artifact metadata"),
        (init_checkpoint, "init checkpoint"),
        (generic_bootstrap, "generic bootstrap replay"),
        (random_teacher, "random teacher replay"),
        (fixed_large_suite, "fixed large suite"),
        (medium_suite, "medium suite"),
    ):
        require_existing_file(path, label)
    for path in heldout_suite_paths:
        require_existing_file(path, f"heldout suite {path.name}")

    input_summary = {
        "current_artifact_weights": verify_expected_hash(
            current_artifact / "weights.json",
            args.expected_current_weights_sha256,
            "current artifact weights",
        ),
        "current_artifact_metadata": build_input_summary(
            current_artifact / "metadata.json"
        ),
        "init_checkpoint": verify_expected_hash(
            init_checkpoint,
            args.expected_init_checkpoint_sha256,
            "init checkpoint",
        ),
        "generic_bootstrap": build_input_summary(generic_bootstrap),
        "random_teacher": build_input_summary(random_teacher),
        "fixed_large_suite": build_input_summary(fixed_large_suite),
        "medium_suite": build_input_summary(medium_suite),
        "heldout_suites": {
            path.stem: build_input_summary(path) for path in heldout_suite_paths
        },
    }

    if disagreement_trace_path.is_file():
        disagreement_analyzed_states = read_jsonl(disagreement_trace_path)
        collection_metadata = {}
    else:
        state_index, collection_metadata = collect_evaluation_states(
            suite_specs=suite_specs,
            artifact_path=current_artifact,
            c_puct=args.c_puct,
            seed=args.seed,
        )
        unique_states = list(state_index.values())
        state_batches = partition_batches(unique_states, args.workers)
        disagreement_analyzed_states = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, min(args.workers, len(state_batches)))
        ) as pool:
            futures = [
                pool.submit(
                    analyze_state_batch,
                    batch=batch,
                    artifact_path=str(current_artifact),
                    simulations=args.simulations,
                    c_puct=args.c_puct,
                    seed=args.seed + batch_index * 1000,
                )
                for batch_index, batch in enumerate(state_batches)
            ]
            for future in futures:
                disagreement_analyzed_states.extend(future.result())
        disagreement_analyzed_states.sort(key=lambda row: str(row["state_hash"]))
        write_jsonl(disagreement_trace_path, disagreement_analyzed_states)

    preliminary_selected_states, disagreement_selection_summary = select_replay_rows(
        disagreement_analyzed_states
    )
    if disagreement_replay_path.is_file():
        disagreement_replay_rows = read_jsonl(disagreement_replay_path)
    else:
        disagreement_continue_batches = partition_batches(
            preliminary_selected_states, args.workers
        )
        disagreement_replay_rows: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, min(args.workers, len(disagreement_continue_batches)))
        ) as pool:
            futures = [
                pool.submit(
                    continue_state_batch,
                    batch=batch,
                    artifact_path=str(current_artifact),
                    simulations=args.simulations,
                    c_puct=args.c_puct,
                    seed=args.seed + 50_000 + batch_index * 1000,
                )
                for batch_index, batch in enumerate(disagreement_continue_batches)
            ]
            for future in futures:
                disagreement_replay_rows.extend(future.result())
        disagreement_replay_rows.sort(key=lambda row: str(row["state_hash"]))
        write_jsonl(disagreement_replay_path, disagreement_replay_rows)

    disagreement_audit, selected_states = disagreement_audit_summary(
        disagreement_analyzed_states,
        collection_metadata,
        disagreement_replay_rows,
    )
    disagreement_audit["tables"]["selection_summary"] = disagreement_selection_summary

    disagreement_hashes = {str(row["state_hash"]) for row in disagreement_replay_rows}
    if (
        stability_trace_path.is_file()
        and stability_selected_path.is_file()
        and stability_replay_path.is_file()
    ):
        stability_candidate_rows = read_jsonl(stability_trace_path)
        stability_selected_rows = read_jsonl(stability_selected_path)
        stability_replay_rows = read_jsonl(stability_replay_path)
        stability_collection_metadata = {}
        stability_selection_summary = {
            "selected_rows": len(stability_selected_rows),
            "unique_rows": len(
                {str(row["state_hash"]) for row in stability_selected_rows}
            ),
        }
    else:
        stability_candidate_rows, stability_collection_metadata = (
            collect_stability_candidates(
                suite_specs=suite_specs,
                artifact_path=current_artifact,
                equal_budgets=[(EQ_768_BUDGET, 768), (EQ_1200_BUDGET, 1200)],
                c_puct=args.c_puct,
                seed=args.seed,
                workers=args.workers,
            )
        )
        stability_candidate_rows.sort(key=lambda row: str(row["state_hash"]))
        write_jsonl(stability_trace_path, stability_candidate_rows)
        stability_selected_rows, stability_selection_summary = select_stability_rows(
            stability_candidate_rows,
            disagreement_hashes,
            target_rows=TARGET_ROWS,
        )
        write_jsonl(stability_selected_path, stability_selected_rows)
        stability_replay_rows = build_stability_replay(
            selected_rows=stability_selected_rows,
            artifact_path=current_artifact,
            c_puct=args.c_puct,
            seed=args.seed,
            workers=args.workers,
        )
        write_jsonl(stability_replay_path, stability_replay_rows)

    puct_audit = build_puct_audit(
        disagreement_analyzed_states=disagreement_analyzed_states,
        disagreement_replay_rows=disagreement_replay_rows,
        disagreement_audit=disagreement_audit,
        stability_candidate_rows=stability_candidate_rows,
        stability_selected_rows=stability_selected_rows,
        stability_replay_rows=stability_replay_rows,
    )
    puct_audit["disagreement"]["selection_summary"] = disagreement_selection_summary
    puct_audit["stability"]["collection_metadata"] = stability_collection_metadata
    puct_audit["stability"]["selection_detail"] = stability_selection_summary
    puct_audit["pr126_comparison"] = build_pr126_comparison(disagreement_audit)
    write_json(workdir / PUCT_AUDIT_FILENAME, puct_audit)

    if (
        puct_audit["disagreement"]["unique_row_count"] < MIN_TRAINABLE_ROWS
        or puct_audit["stability"]["unique_row_count"] < MIN_TRAINABLE_ROWS
    ):
        summary = {
            "schema": SUMMARY_SCHEMA,
            "status": "aborted_before_training",
            "classification": classify_run([], puct_audit)
            if False
            else (
                "current_search_saturated"
                if puct_audit["disagreement"]["unique_row_count"] < MIN_TRAINABLE_ROWS
                or (
                    int(
                        puct_audit["disagreement"]["high_confidence_disagreement_count"]
                    )
                    / max(int(puct_audit["disagreement"]["analyzed_state_count"]), 1)
                    < 0.05
                )
                else "stability_dominated"
            ),
            "inputs": input_summary,
            "puct_audit": puct_audit,
        }
        write_json(workdir / SUMMARY_FILENAME, summary)
        REPORT_PATH.write_text(
            render_report(
                {
                    **summary,
                    "candidates": [],
                    "current_large_mean_384_256": 0.0,
                    "current_large_mean_768_768": 0.0,
                    "current_large_mean_1200_1200": 0.0,
                }
            ),
            encoding="utf-8",
        )
        return 1

    row_counts = {
        "generic_bootstrap": int(input_summary["generic_bootstrap"]["rows"]),
        "random_teacher": int(input_summary["random_teacher"]["rows"]),
        "mined_disagreement": len(disagreement_replay_rows),
        "stability_anchor": len(stability_replay_rows),
    }
    replay_weights = {
        "generic_bootstrap": 4,
        "random_teacher": 1,
        "mined_disagreement": int(args.disagreement_weight),
        "stability_anchor": int(args.stability_weight),
    }
    fractions = effective_sampling_fractions(row_counts, replay_weights)

    candidates: list[dict[str, Any]] = [
        {
            "name": "balanced_current_ref",
            "report_candidate_name": "current",
            "epochs": 0,
            "checkpoint_path": str(init_checkpoint),
            "artifact_dir": str(current_artifact),
            "trainable_scope": "none",
            "train_metrics": None,
        }
    ]
    for epochs in (1, 2):
        name = f"balanced_iter2_w8s4_policy_head_e{epochs}"
        lane_dir = workdir / name
        lane_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_out = lane_dir / "checkpoint.npz"
        epoch_checkpoint_path = lane_dir / f"checkpoint_epoch{epochs}.npz"
        artifact_dir = lane_dir / f"artifact_{name}"
        metrics_path = lane_dir / "train_metrics.json"
        if (
            epoch_checkpoint_path.is_file()
            and (artifact_dir / "weights.json").is_file()
            and metrics_path.is_file()
        ):
            train_metrics = load_json(metrics_path)
        else:
            train_metrics = run_train(
                data_files=(
                    f"{generic_bootstrap},{random_teacher},{disagreement_replay_path},{stability_replay_path}"
                ),
                replay_weights=f"4,1,{args.disagreement_weight},{args.stability_weight}",
                init_checkpoint=str(init_checkpoint),
                out=str(checkpoint_out),
                top_k_dir=str(lane_dir),
                epochs=epochs,
                seed=args.seed,
            )
            export_checkpoint(
                checkpoint_path=str(epoch_checkpoint_path),
                out_dir=str(artifact_dir),
                version=name,
                policy_loss=float(train_metrics.get("policy_loss", 0.0)),
                value_loss=float(train_metrics.get("value_loss", 0.0)),
            )
            write_json(metrics_path, train_metrics)
        candidates.append(
            {
                "name": name,
                "report_candidate_name": artifact_dir.name,
                "epochs": epochs,
                "checkpoint_path": str(epoch_checkpoint_path),
                "artifact_dir": str(artifact_dir),
                "trainable_scope": "policy_head",
                "train_metrics": train_metrics,
                "effective_replay_sampling_fractions": fractions,
            }
        )

    candidate_paths = ",".join(
        str(candidate["artifact_dir"]) for candidate in candidates
    )
    medium_report_path = workdir / "eval_medium" / "temperature_benchmark_report.json"
    if medium_report_path.is_file():
        medium_report = load_json(medium_report_path)
    else:
        medium_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_medium"),
            suite=str(medium_suite),
            current=str(current_artifact),
            candidates=candidate_paths,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )
    fixed_large_report_path = (
        workdir / "eval_fixed_large" / "temperature_benchmark_report.json"
    )
    if fixed_large_report_path.is_file():
        fixed_large_report = load_json(fixed_large_report_path)
    else:
        fixed_large_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_fixed_large"),
            suite=str(fixed_large_suite),
            current=str(current_artifact),
            candidates=candidate_paths,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )
    heldout_reports: dict[str, dict[str, Any]] = {}
    for suite_name, suite_path in suite_specs[1:]:
        heldout_report_path = (
            workdir / f"eval_{suite_name}" / "temperature_benchmark_report.json"
        )
        if heldout_report_path.is_file():
            heldout_reports[suite_name] = load_json(heldout_report_path)
        else:
            heldout_reports[suite_name] = run_opening_suite_benchmark(
                workdir=str(workdir / f"eval_{suite_name}"),
                suite=str(suite_path),
                current=str(current_artifact),
                candidates=candidate_paths,
                budget_pairs=args.budget_pairs,
                games_per_opening=args.games_per_opening,
                seed=args.seed,
                workers=args.workers,
                timeout=args.timeout,
            )

    suite_rows = large_suite_rows(
        reports={"fixed_large": fixed_large_report, **heldout_reports},
        candidates=candidates,
    )
    disagreement_policy_rows = load_candidate_policy_shift_rows(
        disagreement_replay_rows,
        {str(row["state_hash"]): row for row in selected_states},
    )
    stability_policy_rows = load_candidate_policy_shift_rows(
        stability_replay_rows,
        {str(row["state_hash"]): row for row in stability_selected_rows},
    )

    pre_gate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        pre_gate_rows.append(
            build_candidate_summary_row(
                candidate=candidate,
                init_checkpoint=init_checkpoint,
                medium_report=medium_report,
                fixed_large_report=fixed_large_report,
                heldout_reports=heldout_reports,
                suite_rows=suite_rows,
                budget_pairs=budget_pairs,
                disagreement_policy_rows=disagreement_policy_rows,
                stability_policy_rows=stability_policy_rows,
                seed=args.seed,
                gate_report=None,
            )
        )
    gate_targets = gate_targets_from_summary(pre_gate_rows)
    gate_reports: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate["name"] not in gate_targets:
            continue
        gate_report_path = workdir / "gate" / f"{candidate['name']}.json"
        if gate_report_path.is_file():
            gate_reports[str(candidate["name"])] = load_json(gate_report_path)
        else:
            gate_reports[str(candidate["name"])] = run_default_gate(
                candidate_path=str(candidate["artifact_dir"]),
                current_path=str(current_artifact),
                out=str(gate_report_path),
                seed=args.seed,
                workers=args.workers,
            )

    summary_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        summary_candidates.append(
            build_candidate_summary_row(
                candidate=candidate,
                init_checkpoint=init_checkpoint,
                medium_report=medium_report,
                fixed_large_report=fixed_large_report,
                heldout_reports=heldout_reports,
                suite_rows=suite_rows,
                budget_pairs=budget_pairs,
                disagreement_policy_rows=disagreement_policy_rows,
                stability_policy_rows=stability_policy_rows,
                seed=args.seed,
                gate_report=gate_reports.get(str(candidate["name"])),
            )
        )

    current_row = next(
        row for row in summary_candidates if row["candidate"] == "balanced_current_ref"
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "classification": classify_run(summary_candidates, puct_audit),
        "workdir": str(workdir),
        "seed": args.seed,
        "workers": args.workers,
        "budget_pairs": budget_pairs,
        "games_per_opening": args.games_per_opening,
        "inputs": input_summary,
        "guardrails": {
            "promotion": False,
            "overwrite_current": False,
            "architecture_change": False,
            "residual_v4": False,
            "full_network_training": False,
            "last_block_policy_training": False,
            "replay_weight_sweep": False,
            "lr_change": False,
            "classic_mcts_replay": False,
        },
        "replay_row_counts": row_counts,
        "replay_weights": replay_weights,
        "effective_replay_sampling_fractions": fractions,
        "puct_audit": puct_audit,
        "gate_targets": gate_targets,
        "current_large_mean_384_256": float(
            current_row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]
        ),
        "current_large_mean_768_768": float(
            current_row["large_suite_aggregate"][EQ_768_BUDGET]["mean_ds"]
        ),
        "current_large_mean_1200_1200": float(
            current_row["large_suite_aggregate"][EQ_1200_BUDGET]["mean_ds"]
        ),
        "candidates": summary_candidates,
    }
    write_json(workdir / SUMMARY_FILENAME, summary)
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
