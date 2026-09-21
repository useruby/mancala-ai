#!/usr/bin/env python3
"""Run the seed48 final-checkpoint replay-volume triangle without promotion."""

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

SCHEMA = "azlite_seed48_final_checkpoint_volume_triangle_v1"
TRAINING_SEEDS = (443, 1001, 1003, 1009, 1013)
BOOTSTRAP_SEED = 349
METRICS = (
    "arena",
    "raw.optimal_mass",
    "raw.expected_regret",
    "mcts_384.optimal_mass",
    "mcts_384.expected_regret",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def values(cells: list[dict[str, Any]], metric: str) -> list[float]:
    if metric == "arena":
        return [cell["diagnostic_arena"]["score"] for cell in cells]
    group, key = metric.split(".")
    return [cell["exact"][group][key] for cell in cells]


def cells_by_seed(results: dict[str, Any], key: str) -> dict[int, dict[str, Any]]:
    cells = {int(cell["training_seed"]): cell for cell in results[key]}
    if tuple(sorted(cells)) != TRAINING_SEEDS:
        raise ValueError("final_checkpoint_triangle_seed_provenance_invalid")
    return cells


def validate_plan(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if (
        plan.get("schema") != SCHEMA
        or tuple(plan.get("training_seed_group", ())) != TRAINING_SEEDS
    ):
        raise ValueError("final_checkpoint_triangle_plan_invalid")
    if plan.get("replay_weights") != [1, 4, 1, 8, 4]:
        raise ValueError("final_checkpoint_triangle_replay_weights_invalid")
    if any(
        plan.get(flag) is not False
        for flag in (
            "canonical_gate_run",
            "promotion_performed",
            "candidate_selection_performed",
        )
    ):
        raise ValueError("final_checkpoint_triangle_nonpromotion_contract_invalid")
    if (
        sha256_file(ROOT / plan["parent_artifact"] / "weights.json")
        != plan["parent_weights_sha256"]
    ):
        raise ValueError("final_checkpoint_triangle_parent_provenance_invalid")
    if not (ROOT / plan["parent_init_checkpoint"]).is_file():
        raise ValueError("final_checkpoint_triangle_parent_checkpoint_unavailable")
    for source in plan["replay_sources"]:
        if sha256_file(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("final_checkpoint_triangle_replay_provenance_invalid")
    if (
        sha256_file(ROOT / plan["evaluation"]["suite"])
        != plan["evaluation"]["suite_sha256"]
    ):
        raise ValueError("final_checkpoint_triangle_diagnostic_suite_unavailable")
    corpus = json.loads((ROOT / plan["exact_corpus"]).read_text(encoding="utf-8"))
    if corpus.get("scope", {}).get("training_eligible") is not False:
        raise ValueError("final_checkpoint_triangle_exact_corpus_not_evaluation_only")
    for arm in ("a_final", "c_final"):
        config = plan["arms"][arm]
        pool = ROOT / config["pool_path"]
        if (
            not pool.is_file()
            or sha256_file(pool) != config["pool_sha256"]
            or rows(pool) != config["row_count"]
        ):
            raise ValueError(f"final_checkpoint_triangle_{arm}_pool_provenance_invalid")
        training = config["training"]
        if (
            training["final_checkpoint"] != "final"
            or training["max_optimizer_updates"] <= 0
        ):
            raise ValueError(
                f"final_checkpoint_triangle_{arm}_training_contract_invalid"
            )
    b_results = json.loads((ROOT / plan["b_final_results"]).read_text(encoding="utf-8"))
    b_plan = b_results.get("plan", {})
    if b_plan.get("parent_weights_sha256") != plan[
        "parent_weights_sha256"
    ] or b_plan.get("replay_weights") != [1, 4, 1, 8, 4]:
        raise ValueError("final_checkpoint_triangle_b_provenance_invalid")
    if b_plan.get("fixed_pool_sha256") != plan["arms"]["a_final"]["pool_sha256"]:
        raise ValueError("final_checkpoint_triangle_b_pool_provenance_invalid")
    b = cells_by_seed(b_results, "b_fixed71115_high_updates")
    for cell in b.values():
        if cell["training_cost"]["optimizer_updates"] != 3068:
            raise ValueError("final_checkpoint_triangle_b_update_provenance_invalid")
        if cell["training_cost"]["final_checkpoint"] != "final":
            raise ValueError("final_checkpoint_triangle_b_checkpoint_policy_invalid")
        artifact = Path(cell["artifact"])
        if (
            not artifact.is_dir()
            or sha256_file(artifact / "checkpoint.npz") != cell["checkpoint_sha256"]
        ):
            raise ValueError("final_checkpoint_triangle_b_artifact_provenance_invalid")
        if sha256_file(artifact / "weights.json") != cell["weights_sha256"]:
            raise ValueError("final_checkpoint_triangle_b_weights_provenance_invalid")
        arena = cell["diagnostic_arena"]
        suite_sha256 = arena.get("evaluation_config", {}).get("suite_sha256")
        if suite_sha256 is None:
            suite_sha256 = arena["seat_reports"][0]["notes"]["suite_sha256"]
        if arena["games"] != 256 or suite_sha256 != plan["evaluation"]["suite_sha256"]:
            raise ValueError(
                "final_checkpoint_triangle_b_evaluation_provenance_invalid"
            )
    return b


def command_for(
    arm: dict[str, Any], checkpoint: Path, plan: dict[str, Any], seed: int
) -> list[str]:
    command = training_command(
        ROOT / arm["pool_path"], checkpoint, {**plan, "training": arm["training"]}, seed
    )
    command.extend(
        (
            "--max-optimizer-updates",
            str(arm["training"]["max_optimizer_updates"]),
            "--final-checkpoint",
            "final",
        )
    )
    return command


def run_final_determinism_fixture(
    plan: dict[str, Any], workdir: Path
) -> dict[str, Any]:
    """Exercise the final-state export path twice before expensive training."""
    arm = plan["arms"]["a_final"]
    fixture_arm = {**arm, "training": {**arm["training"], "max_optimizer_updates": 8}}
    checkpoints = [workdir / "determinism" / f"run{index}.npz" for index in (1, 2)]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                command_for(fixture_arm, checkpoint, plan, 443), cwd=ROOT, check=True
            )
    hashes = [sha256_file(path) for path in checkpoints]
    return {
        "seed": 443,
        "cap": 8,
        "final_checkpoint": "final",
        "checkpoint_sha256": hashes,
        "identical": hashes[0] == hashes[1],
    }


def cached_arena(artifact: Path, workdir: Path, label: str) -> dict[str, Any] | None:
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


def train_cell(
    arm_name: str, seed: int, plan: dict[str, Any], workdir: Path
) -> dict[str, Any]:
    validate_plan(plan)  # Re-hash every frozen source immediately before each arm.
    arm = plan["arms"][arm_name]
    label = f"{arm_name}-train{seed}"
    artifact = workdir / "cells" / label
    checkpoint, metrics_path, log = (
        artifact / "checkpoint.npz",
        artifact / "training_metrics.json",
        artifact / "train.log",
    )
    artifact.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if not (
        checkpoint.is_file()
        and metrics_path.is_file()
        and (artifact / "weights.json").is_file()
    ):
        command = command_for(arm, checkpoint, plan, seed)
        command.extend(("--training-metrics-out", str(metrics_path)))
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True
            )
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
        duration = time.monotonic() - started
    else:
        duration = None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    training = arm["training"]
    if (
        metrics["optimizer_updates"] != training["max_optimizer_updates"]
        or metrics["final_checkpoint"] != "final"
    ):
        raise ValueError("final_checkpoint_triangle_exact_training_contract_failed")
    exact_path = artifact / "exact.json"
    if exact_path.is_file():
        exact = json.loads(exact_path.read_text(encoding="utf-8"))
    else:
        exact = evaluate_exact(
            artifact,
            {"exact_corpus": {"path": plan["exact_corpus"]}},
            int(plan["evaluation"]["seed"]),
        )
        write_json(exact_path, exact)
    arena = cached_arena(artifact, workdir, label) or run_arena_pair(
        challenger=artifact,
        current=ROOT / plan["parent_artifact"],
        plan={"evaluation": plan["evaluation"]},
        out_dir=workdir / "evaluations",
        label=label,
    )
    regression_path = workdir / "regressions" / f"{label}.json"
    return {
        "training_seed": seed,
        "arm": arm_name,
        "artifact": str(artifact),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights_sha256": sha256_file(artifact / "weights.json"),
        "training_losses": losses(log),
        "training_cost": {
            **metrics,
            "batches_per_completed_epoch": (
                metrics["train_split_count"] + training["batch_size"] - 1
            )
            // training["batch_size"],
            "fresh_row_count": arm["row_count"],
            "wall_clock_training_duration_seconds": duration,
        },
        "exact": exact,
        "diagnostic_arena": arena,
        "regression": json.loads(regression_path.read_text(encoding="utf-8"))
        if regression_path.is_file()
        else run_regressions(artifact, regression_path),
    }


def summary(cells: list[dict[str, Any]], metric: str) -> dict[str, float]:
    items = values(cells, metric)
    return {
        "mean": statistics.fmean(items),
        "median": statistics.median(items),
        "sd": statistics.stdev(items),
        "min": min(items),
        "max": max(items),
        "range": max(items) - min(items),
    }


def paired(
    left: list[dict[str, Any]], right: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    deltas = [
        right_value - left_value
        for left_value, right_value in zip(values(left, metric), values(right, metric))
    ]
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(
        statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(10_000)
    )
    return {
        "by_seed": dict(zip(map(str, TRAINING_SEEDS), deltas)),
        "mean": statistics.fmean(deltas),
        "median": statistics.median(deltas),
        "bootstrap_95": {
            "seed": BOOTSTRAP_SEED,
            "samples": 10_000,
            "lower": means[249],
            "upper": means[9749],
        },
    }


def classify(
    update: dict[str, Any], volume: dict[str, Any], cells: list[list[dict[str, Any]]]
) -> str:
    update_good = update["arena"]["bootstrap_95"]["lower"] > 0.02
    update_bad = update["arena"]["bootstrap_95"]["upper"] < -0.02
    volume_good = (
        volume["arena"]["bootstrap_95"]["lower"] > 0.02
        or volume["raw.optimal_mass"]["bootstrap_95"]["lower"] > 0.01
        or volume["raw.expected_regret"]["bootstrap_95"]["upper"] < -0.1
    )
    regressions_worse = any(not cell["regression"]["passed"] for cell in cells[2])
    if update_bad:
        return "extra_optimizer_work_hurts_small_pool"
    if update_good and volume_good:
        return "optimizer_work_and_data_volume_both_help"
    if volume_good and not regressions_worse:
        return "data_volume_adds_value_beyond_optimizer_work"
    if update_good:
        return "optimizer_work_explains_full_volume_gain"
    return "final_checkpoint_triangle_inconclusive"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-seed48-final-checkpoint-volume-triangle/plan.json",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=ROOT / ".tmp/seed48-final-checkpoint-volume-triangle",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-seed48-final-checkpoint-volume-triangle/results.json",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    try:
        b_by_seed = validate_plan(plan)
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
    if not args.execute:
        write_json(
            args.out,
            {
                "schema": SCHEMA,
                "classification": "execution_pending",
                "plan": plan,
                "canonical_gate_run": False,
                "promotion_performed": False,
                "candidate_selection_performed": False,
            },
        )
        return 0
    fixture = run_final_determinism_fixture(plan, args.workdir)
    if not fixture["identical"]:
        write_json(
            args.out,
            {
                "schema": SCHEMA,
                "classification": "final_checkpoint_reproduction_failed",
                "plan": plan,
                "final_checkpoint_determinism_fixture": fixture,
                "canonical_gate_run": False,
                "promotion_performed": False,
                "candidate_selection_performed": False,
            },
        )
        return 0
    a = [train_cell("a_final", seed, plan, args.workdir) for seed in TRAINING_SEEDS]
    b = [b_by_seed[seed] for seed in TRAINING_SEEDS]
    c = [train_cell("c_final", seed, plan, args.workdir) for seed in TRAINING_SEEDS]
    update, volume = (
        {metric: paired(left, right, metric) for metric in METRICS}
        for left, right in ((a, b), (b, c))
    )
    boundary = {
        name: {
            "batches_per_full_epoch": cells[0]["training_cost"][
                "batches_per_completed_epoch"
            ],
            "full_epochs_completed": cells[0]["training_cost"]["full_epochs_completed"],
            "partial_final_epoch_batches": cells[0]["training_cost"][
                "partial_final_epoch_batches"
            ],
        }
        for name, cells in (("a_final", a), ("c_final", c))
    }
    table = [
        {
            "training_seed": seed,
            "a_final_arena": a_cell["diagnostic_arena"]["score"],
            "b_final_arena": b_cell["diagnostic_arena"]["score"],
            "c_final_arena": c_cell["diagnostic_arena"]["score"],
            "b_minus_a": b_cell["diagnostic_arena"]["score"]
            - a_cell["diagnostic_arena"]["score"],
            "c_minus_b": c_cell["diagnostic_arena"]["score"]
            - b_cell["diagnostic_arena"]["score"],
        }
        for seed, a_cell, b_cell, c_cell in zip(TRAINING_SEEDS, a, b, c)
    ]
    result = {
        "schema": SCHEMA,
        "plan": plan,
        "final_checkpoint_determinism_fixture": fixture,
        "natural_boundary": boundary,
        "a_final": a,
        "b_final": b,
        "c_final": c,
        "required_seed_table": table,
        "summaries": {
            name: {metric: summary(cells, metric) for metric in METRICS}
            for name, cells in (("a_final", a), ("b_final", b), ("c_final", c))
        },
        "update_effect": update,
        "volume_effect": volume,
        "classification": classify(update, volume, [a, b, c]),
        "canonical_gate_run": False,
        "promotion_performed": False,
        "candidate_selection_performed": False,
    }
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
