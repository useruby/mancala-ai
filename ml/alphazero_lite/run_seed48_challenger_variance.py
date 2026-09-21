#!/usr/bin/env python3
"""Measure self-play-data versus training-seed variance without self-play or promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.run_seed48_uniform384_vs_1200 import (  # noqa: E402
    evaluate_exact,
    run_arena_pair,
)

SCHEMA = "azlite_seed48_challenger_variance_v1"
ORIGINAL_WEIGHTS_SHA = (
    "2e517e21097dba818b362aa43b7c6917c17438e1ad0c8a16136ba10a20500a8b"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_inputs(plan: dict[str, Any]) -> None:
    if plan.get("schema") != SCHEMA:
        raise ValueError("variance_audit_plan_schema_invalid")
    if plan["replay_weights"] != [1, 4, 1, 8, 4]:
        raise ValueError("variance_audit_replay_weights_invalid")
    if (
        sha256_file(ROOT / plan["parent_artifact"] / "weights.json")
        != plan["parent_weights_sha256"]
    ):
        raise ValueError("variance_audit_parent_provenance_invalid")
    for source in plan["replay_sources"]:
        if sha256_file(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("variance_audit_replay_provenance_invalid")
    for dataset in plan["selfplay_datasets"]:
        if sha256_file(ROOT / dataset["path"]) != dataset["sha256"]:
            raise ValueError("variance_audit_selfplay_provenance_unavailable")
    if (
        sha256_file(ROOT / plan["evaluation"]["suite"])
        != plan["evaluation"]["suite_sha256"]
    ):
        raise ValueError("variance_audit_diagnostic_suite_unavailable")
    corpus = json.loads((ROOT / plan["exact_corpus"]).read_text(encoding="utf-8"))
    if corpus.get("scope", {}).get("training_eligible") is not False:
        raise ValueError("variance_audit_exact_corpus_not_evaluation_only")


def training_command(
    data: Path, output: Path, plan: dict[str, Any], seed: int
) -> list[str]:
    config = plan["training"]
    replay_paths = [str(ROOT / source["path"]) for source in plan["replay_sources"]]
    return [
        str(ROOT / ".venv/bin/python"),
        "ml/alphazero_lite/train.py",
        "--data",
        str(data),
        "--data-files",
        ",".join([str(data), *replay_paths]),
        "--replay-weights",
        "1,4,1,8,4",
        "--init-checkpoint",
        str(ROOT / plan["parent_init_checkpoint"]),
        "--out",
        str(output),
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--device",
        "auto",
        "--lr-scheduler",
        config["lr_scheduler"],
        "--hidden-sizes",
        config["hidden_sizes"],
        "--model-type",
        config["model_type"],
        "--input-encoding",
        config["input_encoding"],
        "--value-loss",
        config["value_loss"],
        "--huber-delta",
        str(config["huber_delta"]),
        "--value-loss-weight",
        str(config["value_loss_weight"]),
        "--val-split",
        str(config["val_split"]),
        "--grad-clip",
        str(config["grad_clip"]),
        "--save-top-k",
        "3",
        "--policy-target-mode",
        config["policy_target_mode"],
        "--value-target-mode",
        config["value_target_mode"],
        "--seed",
        str(seed),
    ]


def losses(log: Path) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in re.findall(
            r"^(policy_loss|value_loss|total_loss|best_val_loss)=([0-9.]+)$",
            log.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    }


def train_cell(
    label: str, dataset: dict[str, Any], seed: int, plan: dict[str, Any], workdir: Path
) -> dict[str, Any]:
    validate_inputs(plan)  # Hash every replay immediately before each run.
    artifact = workdir / "cells" / label
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact / "checkpoint.npz"
    train_log = artifact / "train.log"
    if not (checkpoint.is_file() and (artifact / "weights.json").is_file()):
        with train_log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                training_command(ROOT / dataset["path"], checkpoint, plan, seed),
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
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
    exact = evaluate_exact(
        artifact,
        {"exact_corpus": {"path": plan["exact_corpus"]}},
        int(plan["evaluation"]["seed"]),
    )
    arena = run_arena_pair(
        challenger=artifact,
        current=ROOT / plan["parent_artifact"],
        plan={"evaluation": plan["evaluation"]},
        out_dir=workdir / "evaluations",
        label=label,
    )
    return {
        "self_play_seed": dataset["seed"],
        "self_play_source": dataset["path"],
        "self_play_sha256": dataset["sha256"],
        "training_seed": seed,
        "parent_weights_sha256": plan["parent_weights_sha256"],
        "replay_sources": plan["replay_sources"],
        "replay_weights": plan["replay_weights"],
        "artifact": str(artifact),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights_sha256": sha256_file(artifact / "weights.json"),
        "training_losses": losses(train_log),
        "exact": exact,
        "diagnostic_arena": arena,
    }


def summarize(cells: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = {
        "diagnostic_arena_score": [cell["diagnostic_arena"]["score"] for cell in cells],
        "raw_optimal_mass": [cell["exact"]["raw"]["optimal_mass"] for cell in cells],
        "raw_expected_regret": [
            cell["exact"]["raw"]["expected_regret"] for cell in cells
        ],
        "mcts_384_optimal_mass": [
            cell["exact"]["mcts_384"]["optimal_mass"] for cell in cells
        ],
        "mcts_384_expected_regret": [
            cell["exact"]["mcts_384"]["expected_regret"] for cell in cells
        ],
    }
    return {
        name: {
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }
        for name, values in metrics.items()
    }


def classify(
    selfplay: dict[str, dict[str, float]], training: dict[str, dict[str, float]]
) -> str:
    left, right = selfplay["diagnostic_arena_score"], training["diagnostic_arena_score"]
    if left["sd"] >= 1.5 * right["sd"] and left["range"] >= 0.05:
        return "selfplay_variance_dominant"
    if right["sd"] >= 1.5 * left["sd"] and right["range"] >= 0.05:
        return "training_variance_dominant"
    if left["range"] >= 0.05 and right["range"] >= 0.05:
        return "both_variance_sources_material"
    return "challenger_variance_small"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "docs/data/alphazero-lite-seed48-challenger-variance/plan.json",
    )
    parser.add_argument(
        "--workdir", type=Path, default=ROOT / ".tmp/seed48-challenger-variance"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-seed48-challenger-variance/results.json",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    try:
        validate_inputs(plan)
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
    phase0_weights = args.workdir / "phase0/weights.json"
    phase0_checkpoint = args.workdir / "phase0/checkpoint.npz"
    phase0 = {
        "attempted": phase0_weights.is_file(),
        "original_weights_sha256": ORIGINAL_WEIGHTS_SHA,
        "rerun_weights_sha256": sha256_file(phase0_weights)
        if phase0_weights.is_file()
        else None,
        "original_checkpoint_sha256": "f65f466ebbfc807a7e561ba383c101dd02f83a40cfc55ab621692ee44dcf2cf5",
        "rerun_checkpoint_sha256": sha256_file(phase0_checkpoint)
        if phase0_checkpoint.is_file()
        else None,
    }
    if phase0["rerun_weights_sha256"] != ORIGINAL_WEIGHTS_SHA:
        write_json(
            args.out,
            {
                "schema": SCHEMA,
                "classification": "training_execution_nondeterministic",
                "phase0": phase0,
                "canonical_gate_run": False,
                "promotion_performed": False,
                "candidate_selection_performed": False,
            },
        )
        return 0
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "phase0": phase0,
        "canonical_gate_run": False,
        "promotion_performed": False,
        "candidate_selection_performed": False,
        "plan": plan,
    }
    if not args.execute:
        result["classification"] = "execution_pending"
        write_json(args.out, result)
        return 0
    datasets = {item["seed"]: item for item in plan["selfplay_datasets"]}
    selfplay_cells = [
        train_cell(
            f"selfplay-s{seed}-train443", datasets[seed], 443, plan, args.workdir
        )
        for seed in (401, 407, 413, 419)
    ]
    training_cells = [
        train_cell(
            f"train-s{seed}-selfplay443", datasets[443], seed, plan, args.workdir
        )
        for seed in plan["training_seed_group"]
        if seed != 443
    ]
    phase0_cell = train_cell(
        "train-s443-selfplay443", datasets[443], 443, plan, args.workdir
    )
    training_cells.insert(0, phase0_cell)
    selfplay_summary, training_summary = (
        summarize(selfplay_cells),
        summarize(training_cells),
    )
    result.update(
        {
            "selfplay_variance_group": {
                "cells": selfplay_cells,
                "summary": selfplay_summary,
            },
            "training_variance_group": {
                "cells": training_cells,
                "summary": training_summary,
            },
            "selfplay_strength_sd": selfplay_summary["diagnostic_arena_score"]["sd"],
            "training_strength_sd": training_summary["diagnostic_arena_score"]["sd"],
            "strength_sd_ratio": selfplay_summary["diagnostic_arena_score"]["sd"]
            / training_summary["diagnostic_arena_score"]["sd"],
            "classification": classify(selfplay_summary, training_summary),
        }
    )
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
