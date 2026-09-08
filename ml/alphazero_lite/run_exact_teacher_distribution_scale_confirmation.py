#!/usr/bin/env python3
"""Run the pre-registered 5,880-row mixed exact-teacher scale confirmation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import exact_teacher_labeling as exact  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import sha256_file  # noqa: E402
from ml.alphazero_lite.run_exact_teacher_distribution_ablation import (  # noqa: E402
    BUDGET_PAIRS,
    SEEDS,
    benchmark_results,
    gate_pass,
    provenance,
    suite_canonical_keys,
    verify_unique,
)
from ml.alphazero_lite.run_exact_teacher_training_ablation import (  # noqa: E402
    export_artifact,
    train_lane,
    verify_holdout_disjoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-train", type=Path, required=True)
    parser.add_argument("--midgame-train", type=Path, required=True)
    parser.add_argument("--opening-holdout", type=Path, required=True)
    parser.add_argument("--midgame-holdout", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--pr283-summary", type=Path, required=True)
    parser.add_argument("--current", type=Path, default=Path("model-artifact/current"))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def select_scaled_mixed(
    opening_rows: list[dict[str, Any]], midgame_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select all 2,940 deterministic rows from each disjoint source cohort."""
    verify_unique(opening_rows, "opening input")
    verify_unique(midgame_rows, "midgame input")
    opening = sorted(
        opening_rows,
        key=lambda row: (str(row["canonical_state"]), str(row["source_id"])),
    )
    midgame = sorted(
        midgame_rows,
        key=lambda row: (str(row["canonical_state"]), str(row["source_id"])),
    )
    if len(opening) < 2940 or len(midgame) < 2940:
        raise ValueError("both inputs must contain at least 2,940 unique rows")
    rows = sorted(
        opening[:2940] + midgame[:2940],
        key=lambda row: (str(row["canonical_state"]), str(row["source_id"])),
    )
    verify_unique(rows, "mixed_50_50_scale")
    return rows


def verify_leakage(
    rows: list[dict[str, Any]], holdouts: list[list[dict[str, Any]]], suite: Path
) -> None:
    keys = {str(row["canonical_state"]) for row in rows}
    if keys & suite_canonical_keys(suite):
        raise ValueError("mixed_50_50_scale overlaps the evaluation suite")
    for holdout in holdouts:
        if keys & {str(row["canonical_state"]) for row in holdout}:
            raise ValueError("mixed_50_50_scale overlaps an exact holdout")


def paired_effect_deltas(
    scaled: dict[str, dict[str, Any]], pr283: dict[str, Any]
) -> dict[str, dict[str, float]]:
    baseline = pr283["lane_results"]["mixed_50_50"]["seeds"]
    return {
        seed: {
            budget: float(scaled[seed]["arena"][budget]["paired_candidate_effect"])
            - float(baseline[seed]["arena"][budget]["paired_candidate_effect"])
            for budget in ("standard", "equal_768", "equal_high", "1200_vs_256")
        }
        for seed in scaled
    }


def compact_arena(arena: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep the committed summary decision-sized; full evidence remains in workdir."""
    return {
        budget: {
            key: data[key]
            for key in (
                "paired_candidate_effect",
                "ds",
                "disadvantaged_seat_score",
                "opening_bootstrap_ci",
            )
        }
        for budget, data in arena.items()
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Exact-Teacher Mixed Distribution Scale Confirmation",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "No checkpoint was promoted. The acceptance gate is unchanged from PR #283.",
        "",
        "## Dataset",
        "",
        f"Lane SHA: `{summary['dataset']['lane_sha256']}`. Rows: 5,880 (2,940 opening, 2,940 midgame).",
        "",
        "## Arena gate",
        "",
        f"Passed seeds: {summary['gate_pass_count']}/3.",
        "",
        "| seed | 384:256 effect (delta vs PR #283) | 768:768 effect (delta) | 1200:1200 effect (delta) | 1200:256 effect (delta) | gate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for seed, run in summary["seeds"].items():
        effects = run["arena"]
        deltas = summary["paired_effect_delta_vs_pr283"][seed]
        cells = [
            f"{effects[budget]['paired_candidate_effect']:+.3f} ({deltas[budget]:+.3f})"
            for budget in ("standard", "equal_768", "equal_high", "1200_vs_256")
        ]
        lines.append(
            f"| {seed} | {' | '.join(cells)} | {'PASS' if run['gate_pass'] else 'fail'} |"
        )
    lines.extend(["", "## Decision", ""])
    if summary["classification"] == "exact_distribution_scale_gate_met":
        lines.append(
            "The scaled mixed lane passed the frozen gate in at least two seeds. Do not promote this experiment."
        )
    else:
        lines.append(
            "The scaled mixed lane did not pass the frozen gate in at least two seeds. Stop exact-teacher training work; do not run a rescue experiment."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    opening = exact.read_jsonl(args.opening_train)
    midgame = exact.read_jsonl(args.midgame_train)
    holdouts = [
        exact.read_jsonl(args.opening_holdout),
        exact.read_jsonl(args.midgame_holdout),
    ]
    rows = select_scaled_mixed(opening, midgame)
    verify_leakage(rows, holdouts, args.suite)
    for holdout in holdouts:
        verify_holdout_disjoint(rows, holdout)
    dataset = args.workdir / "dataset.jsonl"
    exact.write_jsonl(dataset, rows)
    result: dict[str, Any] = {
        "schema": "exact_teacher_distribution_scale_confirmation_v1",
        "promotion": {"performed": False, "reason": "scale_confirmation_only"},
        "pre_registered_success_rule": "mixed_50_50_scale gate passes >=2 of 3 seeds",
        "dataset": {
            "row_count": len(rows),
            "composition": {"opening": 2940, "midgame": 2940},
            "lane_sha256": sha256_file(dataset),
            "exact_teacher_provenance": provenance(rows),
        },
        "training_config": {
            "model_type": "residual_v3",
            "hidden_sizes": [96, 3],
            "input_encoding": "kalah_v3",
            "epochs": 46,
            "lr": 1e-3,
            "batch_size": 512,
            "value_loss_weight": 0.5,
            "lr_scheduler": "cosine",
            "weight_decay": 0.0,
            "init_checkpoint": None,
            "seeds": list(SEEDS),
        },
        "seeds": {},
    }
    candidates: list[str] = []
    for seed in SEEDS:
        lane_dir = args.workdir / "training" / f"seed{seed}"
        checkpoint = lane_dir / "checkpoint.npz"
        artifact = lane_dir / f"artifact_seed{seed}"
        if not checkpoint.is_file():
            train_lane(
                rows,
                model_type="residual_v3",
                hidden_sizes=(96, 3),
                input_encoding="kalah_v3",
                epochs=46,
                batch_size=512,
                lr=1e-3,
                seed=seed,
                value_loss_weight=0.5,
                val_split=0.1,
                device=torch.device(args.device),
                lane_dir=lane_dir,
                init_checkpoint=None,
                lr_scheduler="cosine",
                weight_decay=0.0,
                trainable_scope="all",
            )
        if not (artifact / "weights.json").is_file():
            export_artifact(
                checkpoint,
                artifact,
                model_type="residual_v3",
                input_encoding="kalah_v3",
                version=f"exact-distribution-scale-seed{seed}",
            )
        result["seeds"][f"seed{seed}"] = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "artifact": str(artifact),
        }
        candidates.append(str(artifact))
    command = [
        sys.executable,
        str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
        "--workdir",
        str(args.workdir / "benchmark"),
        "--suite",
        str(args.suite),
        "--current",
        str(args.current),
        "--candidates",
        ",".join(candidates),
        "--budget-pairs",
        BUDGET_PAIRS,
        "--games-per-opening",
        "2",
        "--seed",
        "42",
        "--workers",
        str(args.workers),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    arena = benchmark_results(
        args.workdir / "benchmark" / "temperature_benchmark_report.json"
    )
    for run in result["seeds"].values():
        run["arena"] = arena[Path(run["artifact"]).name]
        run["gate_pass"], run["gate_by_budget"] = gate_pass(run["arena"])
    result["gate_pass_count"] = sum(
        run["gate_pass"] for run in result["seeds"].values()
    )
    result["paired_effect_delta_vs_pr283"] = paired_effect_deltas(
        result["seeds"], json.loads(args.pr283_summary.read_text(encoding="utf-8"))
    )
    result["classification"] = (
        "exact_distribution_scale_gate_met"
        if result["gate_pass_count"] >= 2
        else "exact_distribution_scale_gate_not_met"
    )
    for run in result["seeds"].values():
        run["arena"] = compact_arena(run["arena"])
    result["elapsed_seconds"] = round(time.monotonic() - started, 1)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
