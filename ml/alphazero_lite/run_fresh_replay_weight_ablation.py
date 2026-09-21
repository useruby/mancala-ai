#!/usr/bin/env python3
"""Compare fresh self-play replay weights without self-play or promotion."""

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

from ml.alphazero_lite.run_seed48_challenger_variance import losses  # noqa: E402
from ml.alphazero_lite.run_seed48_uniform384_vs_1200 import (  # noqa: E402
    evaluate_exact,
    run_arena_pair,
    run_regressions,
)
from ml.alphazero_lite.train import load_jsonl_replay  # noqa: E402

SCHEMA = "azlite_fresh_replay_weight_ablation_v1"
SOURCE_SEEDS = (401, 407, 413, 419, 443, 449)
ARMS = {"fresh_w1": [1, 4, 1, 8, 4], "fresh_w4": [4, 4, 1, 8, 4]}
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


def compact_row_count(path: Path, plan: dict[str, Any]) -> int:
    """Use the trainer's loader so count semantics match weighted indexes exactly."""
    training = plan["training"]
    _, _, _, indexes = load_jsonl_replay(
        [path],
        policy_target_mode=training["policy_target_mode"],
        value_target_mode=training["value_target_mode"],
    )
    return int(indexes.size)


def source_accounting(
    dataset: dict[str, Any], plan: dict[str, Any], weights: list[int]
) -> list[dict[str, Any]]:
    sources = [dataset, *plan["replay_sources"]]
    counts = [compact_row_count(ROOT / source["path"], plan) for source in sources]
    weighted = [count * weight for count, weight in zip(counts, weights)]
    total = sum(weighted)
    return [
        {
            "name": "fresh_selfplay"
            if index == 0
            else source.get("name", f"replay_{index}"),
            "path": source["path"],
            "compact_row_count": count,
            "integer_replay_weight": weight,
            "weighted_index_count": weighted_count,
            "weighted_index_fraction": weighted_count / total,
        }
        for index, (source, count, weight, weighted_count) in enumerate(
            zip(sources, counts, weights, weighted)
        )
    ]


def validate_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        plan.get("schema") != SCHEMA
        or tuple(item["seed"] for item in plan.get("selfplay_datasets", []))
        != SOURCE_SEEDS
    ):
        raise ValueError("fresh_replay_weight_plan_invalid")
    if plan.get("arms") != ARMS:
        raise ValueError("fresh_replay_weight_arms_invalid")
    if plan.get("training", {}).get("training_seed") != 443:
        raise ValueError("fresh_replay_weight_training_seed_invalid")
    training = plan["training"]
    if (
        training.get("max_optimizer_updates") != 1084
        or training.get("final_checkpoint") != "final"
    ):
        raise ValueError("fresh_replay_weight_training_contract_invalid")
    if any(
        plan.get(flag) is not False
        for flag in (
            "canonical_gate_run",
            "promotion_performed",
            "candidate_selection_performed",
        )
    ):
        raise ValueError("fresh_replay_weight_nonpromotion_contract_invalid")
    parent = ROOT / plan["parent_artifact"] / "weights.json"
    if not parent.is_file() or sha256_file(parent) != plan["parent_weights_sha256"]:
        raise ValueError("fresh_replay_weight_parent_provenance_invalid")
    if not (ROOT / plan["parent_init_checkpoint"]).is_file():
        raise ValueError("fresh_replay_weight_parent_checkpoint_unavailable")
    for source in plan["replay_sources"]:
        path = ROOT / source["path"]
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ValueError("fresh_replay_weight_replay_provenance_invalid")
    available = []
    for dataset in plan["selfplay_datasets"]:
        path = ROOT / dataset["path"]
        if path.is_file() and sha256_file(path) == dataset["sha256"]:
            available.append(dataset)
    if len(available) < 5:
        raise ValueError("fresh_replay_weight_source_provenance_unavailable")
    suite = ROOT / plan["evaluation"]["suite"]
    if not suite.is_file() or sha256_file(suite) != plan["evaluation"]["suite_sha256"]:
        raise ValueError("fresh_replay_weight_diagnostic_suite_unavailable")
    corpus = json.loads((ROOT / plan["exact_corpus"]).read_text(encoding="utf-8"))
    if corpus.get("scope", {}).get("training_eligible") is not False:
        raise ValueError("fresh_replay_weight_exact_corpus_not_evaluation_only")
    return available


def command_for(
    dataset: dict[str, Any], arm: str, checkpoint: Path, plan: dict[str, Any]
) -> list[str]:
    training = plan["training"]
    data_files = [
        ROOT / dataset["path"],
        *(ROOT / item["path"] for item in plan["replay_sources"]),
    ]
    return [
        str(ROOT / ".venv/bin/python"),
        "ml/alphazero_lite/train.py",
        "--data",
        str(data_files[0]),
        "--data-files",
        ",".join(map(str, data_files)),
        "--replay-weights",
        ",".join(map(str, ARMS[arm])),
        "--init-checkpoint",
        str(ROOT / plan["parent_init_checkpoint"]),
        "--out",
        str(checkpoint),
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
        "--device",
        "auto",
        "--lr-scheduler",
        training["lr_scheduler"],
        "--hidden-sizes",
        training["hidden_sizes"],
        "--model-type",
        training["model_type"],
        "--input-encoding",
        training["input_encoding"],
        "--value-loss",
        training["value_loss"],
        "--huber-delta",
        str(training["huber_delta"]),
        "--value-loss-weight",
        str(training["value_loss_weight"]),
        "--val-split",
        str(training["val_split"]),
        "--grad-clip",
        str(training["grad_clip"]),
        "--save-top-k",
        "3",
        "--policy-target-mode",
        training["policy_target_mode"],
        "--value-target-mode",
        training["value_target_mode"],
        "--seed",
        str(training["training_seed"]),
        "--max-optimizer-updates",
        "1084",
        "--final-checkpoint",
        "final",
    ]


def cached_arena(workdir: Path, label: str) -> dict[str, Any] | None:
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
    dataset: dict[str, Any], arm: str, plan: dict[str, Any], workdir: Path
) -> dict[str, Any]:
    validate_plan(plan)  # Re-hash frozen bytes immediately before every arm.
    label = f"s{dataset['seed']}-{arm}"
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
        command = command_for(dataset, arm, checkpoint, plan) + [
            "--training-metrics-out",
            str(metrics_path),
        ]
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
    if metrics["optimizer_updates"] != 1084 or metrics["final_checkpoint"] != "final":
        raise ValueError("fresh_replay_weight_exact_training_contract_failed")
    exact_path, regression_path = (
        artifact / "exact.json",
        workdir / "regressions" / f"{label}.json",
    )
    exact = (
        json.loads(exact_path.read_text(encoding="utf-8"))
        if exact_path.is_file()
        else evaluate_exact(
            artifact,
            {"exact_corpus": {"path": plan["exact_corpus"]}},
            int(plan["evaluation"]["seed"]),
        )
    )
    if not exact_path.is_file():
        write_json(exact_path, exact)
    arena = cached_arena(workdir, label) or run_arena_pair(
        challenger=artifact,
        current=ROOT / plan["parent_artifact"],
        plan={"evaluation": plan["evaluation"]},
        out_dir=workdir / "evaluations",
        label=label,
    )
    return {
        "self_play_source": dataset,
        "arm": arm,
        "training_seed": 443,
        "parent_weights_sha256": plan["parent_weights_sha256"],
        "replay_weights": ARMS[arm],
        "effective_source_shares": source_accounting(dataset, plan, ARMS[arm]),
        "artifact": str(artifact),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights_sha256": sha256_file(artifact / "weights.json"),
        "training_losses": losses(log),
        "training_cost": {**metrics, "wall_clock_training_duration_seconds": duration},
        "exact": exact,
        "diagnostic_arena": arena,
        "regression": json.loads(regression_path.read_text(encoding="utf-8"))
        if regression_path.is_file()
        else run_regressions(artifact, regression_path),
    }


def metric(cell: dict[str, Any], name: str) -> float:
    if name == "arena":
        return float(cell["diagnostic_arena"]["score"])
    group, key = name.split(".")
    return float(cell["exact"][group][key])


def paired(
    w1: list[dict[str, Any]], w4: list[dict[str, Any]], name: str, seed: int
) -> dict[str, Any]:
    deltas = [metric(right, name) - metric(left, name) for left, right in zip(w1, w4)]
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(10_000)
    )
    return {
        "by_source": {
            str(left["self_play_source"]["seed"]): delta
            for left, delta in zip(w1, deltas)
        },
        "mean": statistics.fmean(deltas),
        "bootstrap_95": {
            "seed": seed,
            "samples": 10_000,
            "lower": samples[249],
            "upper": samples[9749],
        },
    }


def classify(effect: dict[str, Any], w4: list[dict[str, Any]]) -> str:
    raw_mass, raw_regret, arena = (
        effect["raw.optimal_mass"],
        effect["raw.expected_regret"],
        effect["arena"],
    )
    mcts_bad = (
        effect["mcts_384.optimal_mass"]["bootstrap_95"]["upper"] < 0
        or effect["mcts_384.expected_regret"]["bootstrap_95"]["lower"] > 0
    )
    repeated_regressions = sum(not cell["regression"]["passed"] for cell in w4) >= 2
    raw_good = (
        raw_mass["bootstrap_95"]["lower"] > 0
        and raw_regret["bootstrap_95"]["upper"] < 0
    )
    raw_bad = (
        raw_mass["bootstrap_95"]["upper"] < 0
        and raw_regret["bootstrap_95"]["lower"] > 0
    )
    if raw_bad or arena["bootstrap_95"]["upper"] < 0 or repeated_regressions:
        return "fresh_replay_upweighting_hurts"
    if raw_good and arena["mean"] >= 0 and not mcts_bad:
        return "fresh_replay_upweighting_improves_distillation"
    if arena["bootstrap_95"]["lower"] > 0 and not repeated_regressions:
        return "fresh_replay_upweighting_improves_strength_only"
    if raw_good and not mcts_bad:
        return "fresh_replay_upweighting_improves_exact_policy_only"
    return "fresh_replay_upweighting_no_clear_benefit"


def summary(cells: list[dict[str, Any]], name: str) -> dict[str, float]:
    items = [metric(cell, name) for cell in cells]
    return {"mean": statistics.fmean(items), "sd": statistics.stdev(items)}


def required_table(
    w1: list[dict[str, Any]], w4: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "self_play_source": left["self_play_source"]["seed"],
            "w1_fresh_share": left["effective_source_shares"][0][
                "weighted_index_fraction"
            ],
            "w4_fresh_share": right["effective_source_shares"][0][
                "weighted_index_fraction"
            ],
            "w1_arena": metric(left, "arena"),
            "w4_arena": metric(right, "arena"),
            "arena_delta": metric(right, "arena") - metric(left, "arena"),
            "raw_mass_delta": metric(right, "raw.optimal_mass")
            - metric(left, "raw.optimal_mass"),
            "raw_regret_delta": metric(right, "raw.expected_regret")
            - metric(left, "raw.expected_regret"),
            "regressions": {
                "fresh_w1": left["regression"],
                "fresh_w4": right["regression"],
            },
        }
        for left, right in zip(w1, w4)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-fresh-replay-weight-ablation/plan.json",
    )
    parser.add_argument(
        "--workdir", type=Path, default=ROOT / ".tmp/fresh-replay-weight-ablation"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-fresh-replay-weight-ablation/results.json",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    try:
        datasets = validate_plan(plan)
    except (FileNotFoundError, ValueError) as error:
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
    # This is deliberately done before --execute can start an optimizer update.
    accounting_path = args.workdir / "source_replay_accounting.json"
    if accounting_path.is_file():
        accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    else:
        accounting = {
            str(dataset["seed"]): {
                arm: source_accounting(dataset, plan, weights)
                for arm, weights in ARMS.items()
            }
            for dataset in datasets
        }
        write_json(accounting_path, accounting)
    if not args.execute:
        write_json(
            args.out,
            {
                "schema": SCHEMA,
                "classification": "execution_pending",
                "available_source_seeds": [item["seed"] for item in datasets],
                "source_replay_accounting": accounting,
                "plan": plan,
                "canonical_gate_run": False,
                "promotion_performed": False,
                "candidate_selection_performed": False,
            },
        )
        return 0
    w1 = [train_cell(dataset, "fresh_w1", plan, args.workdir) for dataset in datasets]
    w4 = [train_cell(dataset, "fresh_w4", plan, args.workdir) for dataset in datasets]
    effects = {name: paired(w1, w4, name, 351) for name in METRICS}
    write_json(
        args.out,
        {
            "schema": SCHEMA,
            "plan": plan,
            "source_replay_accounting": accounting,
            "fresh_w1": w1,
            "fresh_w4": w4,
            "required_table": required_table(w1, w4),
            "aggregate": {
                "effective_fresh_replay_share": {
                    "fresh_w1": statistics.fmean(
                        cell["effective_source_shares"][0]["weighted_index_fraction"]
                        for cell in w1
                    ),
                    "fresh_w4": statistics.fmean(
                        cell["effective_source_shares"][0]["weighted_index_fraction"]
                        for cell in w4
                    ),
                },
                **{
                    name: {
                        "fresh_w1": summary(w1, name),
                        "fresh_w4": summary(w4, name),
                        "paired_delta": effects[name],
                    }
                    for name in METRICS
                },
            },
            "paired_effects": effects,
            "classification": classify(effects, w4),
            "canonical_gate_run": False,
            "promotion_performed": False,
            "candidate_selection_performed": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
