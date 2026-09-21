#!/usr/bin/env python3
"""Run the non-promoting fixed-pool control matched to PR #347 optimizer work."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.run_seed48_challenger_variance import (  # noqa: E402
    losses,
    training_command,
)
from ml.alphazero_lite.run_seed48_uniform384_vs_1200 import (  # noqa: E402
    evaluate_exact,
    run_arena_pair,
    run_regressions,
)

SCHEMA = "azlite_seed48_update_matched_fixed_volume_v1"
TRAINING_SEEDS = (443, 1001, 1003, 1009, 1013)
BOOTSTRAP_SEED = 348


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def cells_by_seed(results_path: str) -> dict[int, dict[str, Any]]:
    results = json.loads((ROOT / results_path).read_text(encoding="utf-8"))
    cells = results["pooled_cells"]
    by_seed = {int(cell["training_seed"]): cell for cell in cells}
    if tuple(sorted(by_seed)) != TRAINING_SEEDS:
        raise ValueError("update_matched_baseline_seed_provenance_invalid")
    return by_seed


def validate_plan(
    plan: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    if (
        plan.get("schema") != SCHEMA
        or tuple(plan.get("training_seed_group", ())) != TRAINING_SEEDS
    ):
        raise ValueError("update_matched_plan_invalid")
    if plan.get("replay_weights") != [1, 4, 1, 8, 4]:
        raise ValueError("update_matched_replay_weights_invalid")
    if any(
        plan.get(flag) is not False
        for flag in (
            "canonical_gate_run",
            "promotion_performed",
            "candidate_selection_performed",
        )
    ):
        raise ValueError("update_matched_nonpromotion_contract_invalid")
    training = plan["training"]
    if (
        training.get("max_optimizer_updates") != 3068
        or training.get("lr_scheduler") != "none"
        or training.get("final_checkpoint") != "final"
    ):
        raise ValueError("update_matched_training_budget_invalid")
    pool = ROOT / plan["fixed_pool_path"]
    if not pool.is_file() or sha256_file(pool) != plan["fixed_pool_sha256"]:
        raise ValueError("update_matched_fixed_pool_provenance_invalid")
    if (
        sum(1 for line in pool.read_text(encoding="utf-8").splitlines() if line)
        != plan["fixed_pool_row_count"]
    ):
        raise ValueError("update_matched_fixed_pool_row_count_invalid")
    if (
        sha256_file(ROOT / plan["parent_artifact"] / "weights.json")
        != plan["parent_weights_sha256"]
    ):
        raise ValueError("update_matched_parent_provenance_invalid")
    for source in plan["replay_sources"]:
        if sha256_file(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("update_matched_replay_provenance_invalid")
    evaluation = plan["evaluation"]
    if sha256_file(ROOT / evaluation["suite"]) != evaluation["suite_sha256"]:
        raise ValueError("update_matched_diagnostic_suite_unavailable")
    corpus = json.loads((ROOT / plan["exact_corpus"]).read_text(encoding="utf-8"))
    if corpus.get("scope", {}).get("training_eligible") is not False:
        raise ValueError("update_matched_exact_corpus_not_evaluation_only")
    a_results = json.loads(
        (ROOT / plan["baseline_a_results"]).read_text(encoding="utf-8")
    )
    a = cells_by_seed(plan["baseline_a_results"])
    c = cells_by_seed(plan["baseline_c_results"])
    if a_results["plan"]["training"]["epochs"] != 4:
        raise ValueError("update_matched_a_update_provenance_invalid")
    if any(
        cell["training_cost"]["total_optimizer_updates"] != 3068 for cell in c.values()
    ):
        raise ValueError("update_matched_c_update_provenance_invalid")
    return a, c


def cached_arena(
    artifact: Path, plan: dict[str, Any], workdir: Path, label: str
) -> dict[str, Any] | None:
    reports = []
    for seat in (0, 1):
        path = workdir / "evaluations" / f"{label}-starts-{seat}.json"
        if not path.is_file():
            return None
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    wins, draws, losses_count = (
        sum(int(report[key]) for report in reports)
        for key in ("wins", "draws", "losses")
    )
    total = wins + draws + losses_count
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses_count,
        "games": total,
        "score": (wins + 0.5 * draws) / total,
        "seat_reports": reports,
    }


def train_cell(seed: int, plan: dict[str, Any], workdir: Path) -> dict[str, Any]:
    validate_plan(plan)  # Re-hash the frozen inputs immediately before every run.
    label = f"fixed71115-high-updates-train{seed}"
    artifact = workdir / "cells" / label
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint, log, metrics_path = (
        artifact / "checkpoint.npz",
        artifact / "train.log",
        artifact / "training_metrics.json",
    )
    if not (
        checkpoint.is_file()
        and (artifact / "weights.json").is_file()
        and metrics_path.is_file()
    ):
        command = training_command(
            ROOT / plan["fixed_pool_path"], checkpoint, plan, seed
        )
        command.extend(
            [
                "--max-optimizer-updates",
                str(plan["training"]["max_optimizer_updates"]),
                "--final-checkpoint",
                plan["training"]["final_checkpoint"],
                "--training-metrics-out",
                str(metrics_path),
            ]
        )
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True
            )
        duration = time.monotonic() - started
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "ml/alphazero_lite/export_artifact.py",
                "--checkpoint",
                str(checkpoint),
                "--out-dir",
                str(artifact),
                "--version",
                label,
                "--model-type",
                "residual_v3",
                "--rules-version",
                "kalah_v1",
                "--input-encoding",
                "kalah_v3",
            ],
            cwd=ROOT,
            check=True,
        )
    else:
        duration = None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics["optimizer_updates"] != plan["training"]["max_optimizer_updates"]:
        raise ValueError("update_matched_exact_update_requirement_failed")
    arena = cached_arena(artifact, plan, workdir, label)
    regression_path = workdir / "regressions" / f"train{seed}.json"
    return {
        "training_seed": seed,
        "artifact": str(artifact),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights_sha256": sha256_file(artifact / "weights.json"),
        "training_losses": losses(log),
        "training_cost": {
            **metrics,
            "batches_per_completed_epoch": (
                metrics["train_split_count"] + plan["training"]["batch_size"] - 1
            )
            // plan["training"]["batch_size"],
            "compact_replay_row_count": plan["fixed_pool_row_count"]
            + sum(
                sum(
                    1
                    for line in (ROOT / source["path"])
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line
                )
                for source in plan["replay_sources"]
            ),
            "weighted_replay_index_count": metrics["train_split_count"]
            + metrics["validation_count"],
            "effective_average_training_exposures_per_weighted_example": metrics[
                "examples_sampled"
            ]
            / metrics["train_split_count"],
            "wall_clock_training_duration_seconds": duration,
        },
        "exact": evaluate_exact(
            artifact,
            {"exact_corpus": {"path": plan["exact_corpus"]}},
            int(plan["evaluation"]["seed"]),
        ),
        "diagnostic_arena": arena
        or run_arena_pair(
            challenger=artifact,
            current=ROOT / plan["parent_artifact"],
            plan={"evaluation": plan["evaluation"]},
            out_dir=workdir / "evaluations",
            label=label,
        ),
        "regression": json.loads(regression_path.read_text(encoding="utf-8"))
        if regression_path.is_file()
        else run_regressions(artifact, regression_path),
    }


def values(cells: list[dict[str, Any]], metric: str) -> list[float]:
    if metric == "arena":
        return [cell["diagnostic_arena"]["score"] for cell in cells]
    group, key = metric.split(".")
    return [cell["exact"][group][key] for cell in cells]


def summary(cells: list[dict[str, Any]], metric: str) -> dict[str, float]:
    scores = values(cells, metric)
    return {
        "mean": statistics.fmean(scores),
        "sd": statistics.stdev(scores),
        "min": min(scores),
        "max": max(scores),
        "range": max(scores) - min(scores),
    }


def bootstrap(deltas: list[float]) -> dict[str, float | int]:
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(
        statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(10_000)
    )
    return {
        "seed": BOOTSTRAP_SEED,
        "samples": 10_000,
        "lower": means[249],
        "upper": means[9749],
    }


def paired_effect(
    left: list[dict[str, Any]], right: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    deltas = [
        right_value - left_value
        for left_value, right_value in zip(values(left, metric), values(right, metric))
    ]
    return {
        "by_seed": dict(zip(map(str, TRAINING_SEEDS), deltas)),
        "mean": statistics.fmean(deltas),
        "bootstrap_95": bootstrap(deltas),
    }


def classify(
    update: dict[str, dict[str, Any]], volume: dict[str, dict[str, Any]]
) -> str:
    update_good = update["arena"]["bootstrap_95"]["lower"] > 0.02
    update_bad = update["arena"]["bootstrap_95"]["upper"] < -0.02
    volume_good = (
        volume["arena"]["bootstrap_95"]["lower"] > 0.02
        or volume["raw.optimal_mass"]["bootstrap_95"]["lower"] > 0.01
        or volume["raw.expected_regret"]["bootstrap_95"]["upper"] < -0.1
    )
    if update_bad:
        return "extra_optimizer_work_hurts_small_pool"
    if update_good and volume_good:
        return "optimizer_work_and_data_volume_both_help"
    if volume_good:
        return "data_volume_adds_value_beyond_optimizer_work"
    if update_good:
        return "optimizer_work_explains_full_volume_gain"
    return "update_matched_control_inconclusive"


def required_seed_table(
    a: list[dict[str, Any]], b: list[dict[str, Any]], c: list[dict[str, Any]]
) -> list[dict[str, float | int]]:
    return [
        {
            "training_seed": seed,
            "a_arena": a_cell["diagnostic_arena"]["score"],
            "b_arena": b_cell["diagnostic_arena"]["score"],
            "c_arena": c_cell["diagnostic_arena"]["score"],
            "b_minus_a_arena": b_cell["diagnostic_arena"]["score"]
            - a_cell["diagnostic_arena"]["score"],
            "c_minus_b_arena": c_cell["diagnostic_arena"]["score"]
            - b_cell["diagnostic_arena"]["score"],
            "a_raw_optimal_mass": a_cell["exact"]["raw"]["optimal_mass"],
            "b_raw_optimal_mass": b_cell["exact"]["raw"]["optimal_mass"],
            "c_raw_optimal_mass": c_cell["exact"]["raw"]["optimal_mass"],
        }
        for seed, a_cell, b_cell, c_cell in zip(TRAINING_SEEDS, a, b, c)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-seed48-update-matched-fixed-volume/plan.json",
    )
    parser.add_argument(
        "--workdir", type=Path, default=ROOT / ".tmp/seed48-update-matched-fixed-volume"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-seed48-update-matched-fixed-volume/results.json",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    try:
        a_by_seed, c_by_seed = validate_plan(plan)
    except ValueError as error:
        write_json(
            args.out,
            {
                "schema": SCHEMA,
                "classification": str(error),
                "canonical_gate_run": False,
                "promotion_performed": False,
                "candidate_selection_performed": False,
            },
        )
        return 0
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": "execution_pending",
        "plan": plan,
        "canonical_gate_run": False,
        "promotion_performed": False,
        "candidate_selection_performed": False,
    }
    if not args.execute:
        write_json(args.out, result)
        return 0
    a = [a_by_seed[seed] for seed in TRAINING_SEEDS]
    b = [train_cell(seed, plan, args.workdir) for seed in TRAINING_SEEDS]
    c = [c_by_seed[seed] for seed in TRAINING_SEEDS]
    metrics = (
        "arena",
        "raw.optimal_mass",
        "raw.expected_regret",
        "mcts_384.optimal_mass",
        "mcts_384.expected_regret",
    )
    update_effects = {metric: paired_effect(a, b, metric) for metric in metrics}
    volume_effects = {metric: paired_effect(b, c, metric) for metric in metrics}
    result.update(
        {
            "a_fixed71115_low_updates": a,
            "b_fixed71115_high_updates": b,
            "c_full353455_high_updates": c,
            "update_effect": update_effects,
            "volume_effect_at_matched_updates": volume_effects,
            "required_seed_table": required_seed_table(a, b, c),
            "summaries": {
                "a": {metric: summary(a, metric) for metric in metrics},
                "b": {metric: summary(b, metric) for metric in metrics},
                "c": {metric: summary(c, metric) for metric in metrics},
            },
            "classification": classify(update_effects, volume_effects),
        }
    )
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
