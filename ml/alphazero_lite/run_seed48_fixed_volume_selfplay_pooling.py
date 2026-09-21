#!/usr/bin/env python3
"""Run the non-promoting, fixed-row-volume seed48 self-play pooling experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from collections import Counter
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

SCHEMA = "azlite_seed48_fixed_volume_selfplay_pooling_v1"
POOL_SCHEMA = "azlite_fixed_volume_pool_v1"
POOL_SEED = 346
ROWS_PER_SOURCE = 14_223
SOURCE_SEEDS = (401, 407, 413, 419, 443)
TRAINING_SEEDS = (443, 1001, 1003, 1009, 1013)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def canonical_row_bytes(row: dict[str, Any]) -> bytes:
    return json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def selection_key(source_seed: int, row_index: int, row_bytes: bytes) -> str:
    payload = b"|".join(
        (
            str(POOL_SEED).encode(),
            str(source_seed).encode(),
            str(row_index).encode(),
            row_bytes,
        )
    )
    return hashlib.sha256(payload).hexdigest()


def final_shuffle_key(source_seed: int, row_index: int, row_bytes: bytes) -> str:
    payload = b"|".join(
        (
            b"final",
            str(POOL_SEED).encode(),
            str(source_seed).encode(),
            str(row_index).encode(),
            row_bytes,
        )
    )
    return hashlib.sha256(payload).hexdigest()


def source_rows(source: dict[str, Any]) -> list[tuple[int, bytes, dict[str, Any]]]:
    path = ROOT / source["path"]
    if not path.is_file() or sha256_file(path) != source["sha256"]:
        raise FileNotFoundError("pooled_selfplay_source_provenance_unavailable")
    rows = [
        (index, canonical_row_bytes(row := json.loads(line)), row)
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
        if line
    ]
    if len(rows) < ROWS_PER_SOURCE:
        raise FileNotFoundError("pooled_selfplay_source_provenance_unavailable")
    return rows


def state_key(row: dict[str, Any]) -> str:
    return json.dumps(row["state"], separators=(",", ":"), ensure_ascii=True)


def bucket(row: dict[str, Any]) -> str:
    ply = int(row.get("move_index", -1))
    if ply < 0:
        return "metadata_unavailable"
    if ply < 10:
        return "ply_0_9"
    if ply < 30:
        return "ply_10_29"
    return "ply_30_plus"


def construct_pool(plan: dict[str, Any], output: Path) -> dict[str, Any]:
    sources = plan.get("selfplay_sources", [])
    if tuple(source.get("seed") for source in sources) != SOURCE_SEEDS:
        raise ValueError("pooled_selfplay_source_provenance_unavailable")
    selected: list[tuple[int, int, bytes, dict[str, Any]]] = []
    source_manifest = []
    source_states: dict[int, set[str]] = {}
    source_buckets: dict[int, Counter[str]] = {}
    all_states: Counter[str] = Counter()
    for source in sources:
        rows = source_rows(source)
        source_seed = int(source["seed"])
        ranked = sorted(
            rows,
            key=lambda item: (selection_key(source_seed, item[0], item[1]), item[0]),
        )[:ROWS_PER_SOURCE]
        subset_bytes = b"".join(row_bytes + b"\n" for _, row_bytes, _ in ranked)
        source_manifest.append(
            {
                "seed": source_seed,
                "source_path": source["path"],
                "source_sha256": source["sha256"],
                "source_row_count": len(rows),
                "selected_row_count": len(ranked),
                "selected_subset_sha256": hashlib.sha256(subset_bytes).hexdigest(),
            }
        )
        source_states[source_seed] = {state_key(row) for _, _, row in ranked}
        source_buckets[source_seed] = Counter(bucket(row) for _, _, row in ranked)
        all_states.update(state_key(row) for _, _, row in ranked)
        selected.extend(
            (source_seed, index, row_bytes, row) for index, row_bytes, row in ranked
        )
    shuffled = sorted(
        selected,
        key=lambda item: (
            final_shuffle_key(item[0], item[1], item[2]),
            item[0],
            item[1],
        ),
    )
    pool_bytes = b"".join(row_bytes + b"\n" for _, _, row_bytes, _ in shuffled)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(pool_bytes)
    shared = set()
    for left_index, left_seed in enumerate(SOURCE_SEEDS):
        for right_seed in SOURCE_SEEDS[left_index + 1 :]:
            shared.update(source_states[left_seed] & source_states[right_seed])
    return {
        "schema": POOL_SCHEMA,
        "pool_seed": POOL_SEED,
        "sampling_algorithm": "sha256(pool_seed|source_seed|row_index|canonical_row_bytes), lowest per source; sha256(final|pool_seed|source_seed|row_index|canonical_row_bytes) final shuffle",
        "pool_path": str(output.relative_to(ROOT)),
        "pool_sha256": hashlib.sha256(pool_bytes).hexdigest(),
        "pool_row_count": len(shuffled),
        "sources": source_manifest,
        "diagnostics": {
            "unique_canonical_state_count": len(all_states),
            "duplicate_state_rate": 1 - (len(all_states) / len(shuffled)),
            "states_shared_across_more_than_one_source": len(shared),
            "source_contribution_by_ply_bucket": {
                str(seed): dict(sorted(source_buckets[seed].items()))
                for seed in SOURCE_SEEDS
            },
        },
    }


def validate_plan(plan: dict[str, Any]) -> None:
    if (
        plan.get("schema") != SCHEMA
        or tuple(plan.get("training_seed_group", ())) != TRAINING_SEEDS
    ):
        raise ValueError("pooled_selfplay_plan_invalid")
    if plan.get("replay_weights") != [1, 4, 1, 8, 4]:
        raise ValueError("pooled_selfplay_replay_weights_invalid")
    if any(
        plan.get(flag) is not False
        for flag in (
            "canonical_gate_run",
            "promotion_performed",
            "candidate_selection_performed",
        )
    ):
        raise ValueError("pooled_selfplay_nonpromotion_contract_invalid")
    if (
        sha256_file(ROOT / plan["parent_artifact"] / "weights.json")
        != plan["parent_weights_sha256"]
    ):
        raise ValueError("pooled_selfplay_parent_provenance_invalid")
    for source in plan["replay_sources"]:
        if sha256_file(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("pooled_selfplay_replay_provenance_invalid")
    if (
        sha256_file(ROOT / plan["evaluation"]["suite"])
        != plan["evaluation"]["suite_sha256"]
    ):
        raise ValueError("pooled_selfplay_diagnostic_suite_unavailable")
    baseline_plan = json.loads(
        (ROOT / plan["baseline_results"]).read_text(encoding="utf-8")
    )["plan"]
    contract_fields = (
        "parent_artifact",
        "parent_weights_sha256",
        "parent_init_checkpoint",
        "training",
        "replay_weights",
        "replay_sources",
        "training_seed_group",
        "exact_corpus",
        "evaluation",
    )
    if any(plan[field] != baseline_plan[field] for field in contract_fields):
        raise ValueError("pooled_selfplay_training_contract_invalid")


def baseline_cells(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
    results = json.loads((ROOT / plan["baseline_results"]).read_text(encoding="utf-8"))
    cells = results["training_variance_group"]["cells"]
    by_seed = {int(cell["training_seed"]): cell for cell in cells}
    if tuple(sorted(by_seed)) != TRAINING_SEEDS or any(
        cell["self_play_sha256"] != plan["single443_sha256"]
        for cell in by_seed.values()
    ):
        raise ValueError("pooled_selfplay_baseline_provenance_invalid")
    return by_seed


def train_pooled(
    seed: int, pool_path: Path, plan: dict[str, Any], workdir: Path
) -> dict[str, Any]:
    artifact = workdir / "cells" / f"pooled5-fixed71115-train{seed}"
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint, log = artifact / "checkpoint.npz", artifact / "train.log"
    if not (checkpoint.is_file() and (artifact / "weights.json").is_file()):
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                training_command(pool_path, checkpoint, plan, seed),
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
                f"pooled5-fixed71115-train{seed}",
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
    return {
        "training_seed": seed,
        "artifact": str(artifact),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights_sha256": sha256_file(artifact / "weights.json"),
        "training_losses": losses(log),
        "exact": evaluate_exact(
            artifact,
            {"exact_corpus": {"path": plan["exact_corpus"]}},
            int(plan["evaluation"]["seed"]),
        ),
        "diagnostic_arena": run_arena_pair(
            challenger=artifact,
            current=ROOT / plan["parent_artifact"],
            plan={"evaluation": plan["evaluation"]},
            out_dir=workdir / "evaluations",
            label=f"pooled5-train{seed}",
        ),
        "regression": run_regressions(
            artifact, workdir / "regressions" / f"train{seed}.json"
        ),
    }


def summary(cells: list[dict[str, Any]], metric: str) -> dict[str, float]:
    values = [
        cell["diagnostic_arena"]["score"]
        if metric == "arena"
        else cell["exact"][metric.split(".")[0]][metric.split(".")[1]]
        for cell in cells
    ]
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def bootstrap(deltas: list[float]) -> dict[str, float | int]:
    rng = random.Random(POOL_SEED)
    means = sorted(
        statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(10_000)
    )
    return {
        "seed": POOL_SEED,
        "samples": 10_000,
        "lower": means[249],
        "upper": means[9749],
    }


def paired_exact_deltas(
    single: list[dict[str, Any]], pooled: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    metrics = (
        ("raw_optimal_mass", "raw", "optimal_mass"),
        ("raw_expected_regret", "raw", "expected_regret"),
        ("mcts_384_optimal_mass", "mcts_384", "optimal_mass"),
        ("mcts_384_expected_regret", "mcts_384", "expected_regret"),
    )
    return {
        name: {
            str(seed): pooled_cell["exact"][group][metric]
            - single_cell["exact"][group][metric]
            for seed, single_cell, pooled_cell in zip(TRAINING_SEEDS, single, pooled)
        }
        for name, group, metric in metrics
    }


def classify(
    arena_deltas: list[float],
    interval: dict[str, float | int],
    single: list[dict[str, Any]],
    pooled: list[dict[str, Any]],
) -> str:
    single_sd, pooled_sd = (
        summary(single, "arena")["sd"],
        summary(pooled, "arena")["sd"],
    )
    exact_mass = statistics.fmean(
        p["exact"]["raw"]["optimal_mass"] - s["exact"]["raw"]["optimal_mass"]
        for s, p in zip(single, pooled)
    )
    exact_regret = statistics.fmean(
        p["exact"]["raw"]["expected_regret"] - s["exact"]["raw"]["expected_regret"]
        for s, p in zip(single, pooled)
    )
    regression_failures = sum(not cell["regression"]["passed"] for cell in pooled)
    if (
        interval["upper"] < 0
        or exact_mass < -0.01
        or exact_regret > 0.1
        or regression_failures >= 3
    ):
        return "fixed_volume_pooling_hurts"
    stronger = interval["lower"] > 0
    stable = pooled_sd <= 0.75 * single_sd
    if stronger and stable:
        return "fixed_volume_pooling_improves_stability_and_strength"
    if stable:
        return "fixed_volume_pooling_improves_stability_only"
    if stronger:
        return "fixed_volume_pooling_improves_strength_only"
    return "fixed_volume_pooling_no_clear_benefit"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-seed48-fixed-volume-selfplay-pooling/plan.json",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=ROOT / ".tmp/seed48-fixed-volume-selfplay-pooling",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-seed48-fixed-volume-selfplay-pooling/results.json",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize:
        result = json.loads(args.out.read_text(encoding="utf-8"))
        result["paired_exact_deltas"] = paired_exact_deltas(
            result["single443_cells"], result["pooled_cells"]
        )
        write_json(args.out, result)
        return 0
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    try:
        validate_plan(plan)
        manifest = construct_pool(plan, args.workdir / "pooled5_fixed71115.jsonl")
        baseline = baseline_cells(plan)
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
    write_json(args.out.parent / "pool_manifest.json", manifest)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": "execution_pending",
        "plan": plan,
        "pool_manifest": manifest,
        "canonical_gate_run": False,
        "promotion_performed": False,
        "candidate_selection_performed": False,
    }
    if not args.execute:
        write_json(args.out, result)
        return 0
    pooled = [
        train_pooled(
            seed, args.workdir / "pooled5_fixed71115.jsonl", plan, args.workdir
        )
        for seed in TRAINING_SEEDS
    ]
    single = [baseline[seed] for seed in TRAINING_SEEDS]
    deltas = [
        p["diagnostic_arena"]["score"] - s["diagnostic_arena"]["score"]
        for s, p in zip(single, pooled)
    ]
    interval = bootstrap(deltas)
    result.update(
        {
            "single443_cells": single,
            "pooled_cells": pooled,
            "paired_arena_deltas": dict(zip(map(str, TRAINING_SEEDS), deltas)),
            "paired_arena_delta_mean": statistics.fmean(deltas),
            "paired_arena_delta_median": statistics.median(deltas),
            "paired_bootstrap_95": interval,
            "paired_exact_deltas": paired_exact_deltas(single, pooled),
            "single443_summary": {
                "arena": summary(single, "arena"),
                "raw_optimal_mass": summary(single, "raw.optimal_mass"),
                "raw_expected_regret": summary(single, "raw.expected_regret"),
                "mcts_384_optimal_mass": summary(single, "mcts_384.optimal_mass"),
                "mcts_384_expected_regret": summary(single, "mcts_384.expected_regret"),
            },
            "pooled_summary": {
                "arena": summary(pooled, "arena"),
                "raw_optimal_mass": summary(pooled, "raw.optimal_mass"),
                "raw_expected_regret": summary(pooled, "raw.expected_regret"),
                "mcts_384_optimal_mass": summary(pooled, "mcts_384.optimal_mass"),
                "mcts_384_expected_regret": summary(pooled, "mcts_384.expected_regret"),
            },
            "classification": classify(deltas, interval, single, pooled),
        }
    )
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
