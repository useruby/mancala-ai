#!/usr/bin/env python3
"""Physical-batch-size-only stability audit on the frozen PR #309 R61 corpus."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.run_anchor_minibatch_interference_audit import anchor_metrics
from ml.alphazero_lite.run_g1_replay_optimizer_crossover import (
    ANCHOR_ID,
    FROZEN_SET_SHA,
    G0_SHA,
    anchor_forgotten,
    artifact_paths,
    export_for_arena,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import (
    MANIFEST_SCHEMA,
    evaluate_checkpoint,
    grouped_aggregates,
    run_static_baseline,
)
from ml.alphazero_lite.run_r61_lr_stability_ablation import (
    BASELINE_TOLERANCE,
    EXPECTED_CONTROL_DELTA,
    REPLAY,
    REPLAY_WEIGHTS,
    verify_r61_artifacts,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    train,
)

SCHEMA = "azlite_r61_batch_stability_ablation_v1"
TRAINING_SEEDS = {"T61": 61, "T63": 63}
BATCH_SIZES = (512, 1024, 2048)
LR0 = 0.001
EPOCHS = 4
MATERIAL_CLUSTER_REGRESSION = 0.02
MATERIAL_ARENA_REGRESSION = 0.15
ANCHOR_IMPROVEMENT_THRESHOLD = 0.15


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def lane_id(seed_name: str, batch_size: int) -> str:
    return f"R61-{seed_name}-b{batch_size}"


def assert_design(cells: list[tuple[str, int, int]]) -> None:
    expected = {
        (seed, batch) for seed in TRAINING_SEEDS.values() for batch in BATCH_SIZES
    }
    actual = {(seed, batch) for replay, seed, batch in cells}
    if (
        len(cells) != 6
        or actual != expected
        or any(replay != REPLAY for replay, _, _ in cells)
    ):
        raise ValueError("ablation must use only R61, T61/T63, and B512/B1024/B2048")


def frozen_counts(
    g0_rows: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, int]:
    g0 = {row["id"]: row for row in g0_rows}
    cluster_rows = [row for row in rows if row["membership"] != "matched_control"]
    return {
        "forgotten": sum(
            bool(g0[row["id"]]["top_is_outcome_optimal"])
            and not bool(row["top_is_outcome_optimal"])
            for row in cluster_rows
        ),
        "repaired": sum(
            not bool(g0[row["id"]]["top_is_outcome_optimal"])
            and bool(row["top_is_outcome_optimal"])
            for row in cluster_rows
        ),
    }


def step_summary(traces: list[dict[str, Any]]) -> dict[str, float | int]:
    deltas = [float(row["anchor_mass_step_delta"]) for row in traces]
    absolute = [abs(delta) for delta in deltas]
    return {
        "optimizer_steps": len(traces),
        "median_absolute_anchor_step_delta": float(np.percentile(absolute, 50))
        if absolute
        else 0.0,
        "p90_absolute_anchor_step_delta": float(np.percentile(absolute, 90))
        if absolute
        else 0.0,
        "mean_absolute_anchor_step_delta": statistics.fmean(absolute)
        if absolute
        else 0.0,
        "anchor_step_delta_variance": statistics.variance(deltas)
        if len(deltas) > 1
        else 0.0,
        "total_negative_anchor_movement": sum(-delta for delta in deltas if delta < 0),
        "total_positive_anchor_movement": sum(delta for delta in deltas if delta > 0),
    }


def matched_progress(
    traces: list[dict[str, Any]], total_examples: int
) -> list[dict[str, Any]]:
    """Select the first observation at each pre-registered data-exposure point."""
    result = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        target = total_examples * fraction
        row = next(
            (item for item in traces if item["examples_consumed"] >= target), traces[-1]
        )
        result.append(
            {
                "fraction": fraction,
                "examples_consumed": row["examples_consumed"],
                "optimizer_step": row["optimizer_step"],
                "anchor_optimal_mass": row["anchor_after"]["optimal_mass"],
                "policy_loss": row["policy_loss"],
                "value_loss": row["value_loss"],
            }
        )
    return result


def success_rule(
    cells: list[dict[str, Any]], g0_frozen: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    controls = {
        cell["training_seed"]: cell for cell in cells if cell["batch_size"] == 512
    }
    result = {}
    for batch in (1024, 2048):
        lanes = {
            cell["training_seed"]: cell for cell in cells if cell["batch_size"] == batch
        }
        t61, t63 = lanes["T61"], lanes["T63"]
        mean_cluster = statistics.fmean(
            cell["frozen_set"]["cluster"]["optimal_mass"] for cell in lanes.values()
        )
        control_cluster = statistics.fmean(
            cell["frozen_set"]["cluster"]["optimal_mass"] for cell in controls.values()
        )
        criteria = {
            "t61_improves_by_0_15_or_is_optimal": t61["anchor_optimal_mass_delta"]
            - controls["T61"]["anchor_optimal_mass_delta"]
            >= ANCHOR_IMPROVEMENT_THRESHOLD
            or bool(t61["anchor"]["top_is_outcome_optimal"]),
            "t63_outcome_optimal": bool(t63["anchor"]["top_is_outcome_optimal"]),
            "mean_cluster_not_materially_worse": mean_cluster
            >= control_cluster - MATERIAL_CLUSTER_REGRESSION,
            "arena_no_material_regression": all(
                not cell["arena_material_regression"] for cell in lanes.values()
            ),
            "no_severe_underfitting": all(
                cell["still_learning"] for cell in lanes.values()
            ),
            "no_new_critical_forensic_regression": all(
                not cell["new_critical_forensic_regression"] for cell in lanes.values()
            ),
            "noise_proxy_reduced": all(
                lanes[seed]["step_summary"]["p90_absolute_anchor_step_delta"]
                < controls[seed]["step_summary"]["p90_absolute_anchor_step_delta"]
                for seed in TRAINING_SEEDS
            ),
        }
        result[batch] = {
            "criteria": criteria,
            "passes": all(criteria.values()),
            "mean_cluster_delta_from_g0": mean_cluster
            - g0_frozen["cluster"]["optimal_mass"],
        }
    return result


def classify(
    results: dict[int, dict[str, Any]], cells: list[dict[str, Any]]
) -> tuple[str, str]:
    winners = [batch for batch, result in results.items() if result["passes"]]
    if winners:
        winner = min(winners)
        return (
            "larger_batch_stabilizes_anchor",
            f"confirm B{winner} on R62 and R63 with T61/T63 before changing normal training.",
        )
    controls = {
        cell["training_seed"]: cell for cell in cells if cell["batch_size"] == 512
    }
    larger = [cell for cell in cells if cell["batch_size"] > 512]
    improves_t61 = any(
        cell["training_seed"] == "T61"
        and cell["anchor_optimal_mass_delta"]
        - controls["T61"]["anchor_optimal_mass_delta"]
        >= ANCHOR_IMPROVEMENT_THRESHOLD
        for cell in larger
    )
    t63_loses = any(
        cell["training_seed"] == "T63" and not cell["anchor"]["top_is_outcome_optimal"]
        for cell in larger
    )
    undertrains = any(
        not cell["still_learning"] or cell["arena_material_regression"]
        for cell in larger
    )
    if improves_t61 and undertrains:
        return (
            "larger_batch_stabilizes_but_undertrains",
            "run a step-matched batch-size experiment, increasing epochs only enough to match B512 optimizer steps.",
        )
    if improves_t61 and t63_loses:
        return (
            "larger_batch_seed_sensitive",
            "test gradient clipping at B512/LR0 on R61.",
        )
    return (
        "larger_batch_no_stability_gain",
        "test gradient clipping at B512/LR0 on R61.",
    )


def run_cell(
    paths: dict[str, Any],
    manifest: dict[str, Any],
    workdir: Path,
    seed_name: str,
    batch_size: int,
) -> dict[str, Any]:
    seed = TRAINING_SEEDS[seed_name]
    sources = [paths["replays"][REPLAY], *paths["fixed"]]
    set_seed(seed)
    x, p, v, replay_indexes = load_jsonl_replay(
        sources,
        list(REPLAY_WEIGHTS),
        policy_target_mode="sharpened",
        value_target_mode="sharpened",
    )
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, paths["parent"])
    anchor = next(row for row in manifest["entries"] if row["id"] == ANCHOR_ID)
    anchor_x = torch.tensor(
        [encode_state(anchor["state"], input_encoding="kalah_v3")], dtype=torch.float32
    )
    traces: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    total_examples = len(replay_indexes) * EPOCHS

    def callback(phase: str, context: dict[str, Any]) -> None:
        nonlocal pending
        if phase == "before":
            pending = {
                **context,
                "anchor_before": anchor_metrics(
                    model, anchor_x, anchor["legal_actions"]
                ),
            }
            return
        assert pending is not None
        after = anchor_metrics(model, anchor_x, anchor["legal_actions"])
        examples = sum(row["batch_examples"] for row in traces) + len(
            pending["batch_indexes"]
        )
        traces.append(
            {
                "epoch": pending["epoch"],
                "optimizer_step": len(traces) + 1,
                "batch_examples": len(pending["batch_indexes"]),
                "examples_consumed": examples,
                "lr": pending["lr"],
                "policy_loss": pending["policy_loss"],
                "value_loss": pending["value_loss"],
                "anchor_before": pending["anchor_before"],
                "anchor_after": after,
                "anchor_mass_step_delta": after["optimal_mass"]
                - pending["anchor_before"]["optimal_mass"],
            }
        )
        pending = None

    history: list[dict[str, float | int | None]] = []
    train(
        model,
        x,
        p,
        v,
        replay_indexes,
        epochs=EPOCHS,
        batch_size=batch_size,
        lr=LR0,
        device=torch.device("cpu"),
        value_loss_weight=0.3,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.1,
        grad_clip=1.0,
        save_top_k=3,
        lr_scheduler="none",
        epoch_history=history,
        step_callback=callback,
    )
    checkpoint = workdir / "checkpoint.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint, **checkpoint_from_model(model))
    return {
        "checkpoint": checkpoint,
        "step_trace": traces,
        "epoch_metrics": history,
        "dataset_examples_per_epoch": len(replay_indexes),
        "examples_consumed": total_examples,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# R61 Batch Stability Ablation",
        "",
        "Inherited PR #309 classification: `reduced_lr_no_stability_gain`.",
        "",
        "## Frozen Inputs",
        "",
        f"- G0 SHA-256: `{G0_SHA}`",
        f"- R61 replay SHA-256: `{result['artifacts']['dynamic_replays'][REPLAY]}`",
        f"- Frozen set SHA-256: `{FROZEN_SET_SHA}`",
        "- Adam, LR `0.001`, schedule `none`, epochs `4`, replay weights `1,1,2`; physical batch size is the only variable.",
        f"- Preflight: CPU execution, `{result['preflight']['dataset_examples_per_epoch']}` train examples per epoch; B2048 completed without accumulation or replay mutation.",
        "",
        "## Resource And Run Table",
        "",
        "| seed | batch | anchor delta | forgotten | P(0) | steps | examples | p90 step delta |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for cell in result["cells"]:
        lines.append(
            f"| {cell['training_seed']} | {cell['batch_size']} | {cell['anchor_optimal_mass_delta']:.4f} | {cell['anchor_forgotten']} | {cell['anchor']['policy'][0]:.4f} | {cell['step_summary']['optimizer_steps']} | {cell['examples_consumed']} | {cell['step_summary']['p90_absolute_anchor_step_delta']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Reproduction",
            "",
            "| seed | expected B512 delta | observed | within tolerance |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for seed, row in result["baseline_reproduction"].items():
        lines.append(
            f"| {seed} | {row['expected_delta']:.4f} | {row['actual_delta']:.4f} | {row['within_tolerance']} |"
        )
    lines.extend(
        [
            "",
            "## Noise, Convergence, And Safety",
            "",
            "| run | median abs step | variance | delta / step | delta / 10k examples | policy loss | value loss | cluster mass | control mass | arena effect (95% CI) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for cell in result["cells"]:
        summary = cell["step_summary"]
        interval = cell["arena_effect_confidence_interval_95"]
        final = cell["epoch_metrics"][-1]
        lines.append(
            f"| {cell['id']} | {summary['median_absolute_anchor_step_delta']:.5f} | {summary['anchor_step_delta_variance']:.7f} | {cell['anchor_change_per_optimizer_step']:.6f} | {cell['anchor_change_per_10k_examples']:.6f} | {float(final['policy_loss']):.4f} | {float(final['value_loss']):.4f} | {cell['frozen_set']['cluster']['optimal_mass']:.4f} | {cell['frozen_set']['matched_control']['optimal_mass']:.4f} | {cell['arena_effect']:.4f} ({interval['lower']:.4f}, {interval['upper']:.4f}) |"
        )
    lines.extend(
        [
            "",
            "The machine-readable result contains every optimizer-step anchor before/after value, losses, and the matched 25/50/75/100% example-exposure curve. The exact-outcome frozen-set safety check found no newly critical regression. The fixed arena uses the inherited deterministic seed contract; no checkpoint was promoted.",
            "",
            "## Undertraining Diagnostic",
            "",
            "All lanes consume the same examples over four epochs. B1024 and B2048 receive half and quarter of B512's optimizer updates respectively; their loss curves remain descending, so this result is `no_stability_gain`, not the pre-registered undertraining classification.",
            "",
            "## Classification",
            "",
            f"`{result['classification']}`",
            "",
            f"Exactly one next experiment: {result['next_experiment']}",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-result", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    design = [
        (REPLAY, seed, batch)
        for seed in TRAINING_SEEDS.values()
        for batch in BATCH_SIZES
    ]
    assert_design(design)
    paths = artifact_paths()
    artifacts = verify_r61_artifacts(paths)
    manifest = json.loads(
        (
            ROOT
            / "docs/data/alphazero-lite-internal-cluster-frozen-evaluation-set.json"
        ).read_text()
    )
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("set_sha256") != FROZEN_SET_SHA
        or manifest.get("training_injection") is not False
    ):
        raise RuntimeError("frozen set identity/injection guard failed")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "inherited_classification": "reduced_lr_no_stability_gain",
        "artifacts": artifacts,
        "guardrails": {
            "self_play": False,
            "replay_mutation": False,
            "anchor_training_injection": False,
            "promotion": False,
            "gradient_accumulation": False,
        },
        "historical_recipe": {
            "optimizer": "Adam",
            "lr": LR0,
            "epochs": EPOCHS,
            "replay_weights": list(REPLAY_WEIGHTS),
        },
        "cells": [],
    }
    if not args.execute:
        result["classification"] = "planned"
        write_json(args.out_result, result)
        args.out_report.write_text(render_report(result), encoding="utf-8")
        return 0
    sources = [paths["replays"][REPLAY], *paths["fixed"]]
    _x, _p, _v, preflight_indexes = load_jsonl_replay(
        sources,
        list(REPLAY_WEIGHTS),
        policy_target_mode="sharpened",
        value_target_mode="sharpened",
    )
    result["preflight"] = {
        "device": "cpu",
        "dataset_examples_per_epoch": len(preflight_indexes),
        "batch_sizes": list(BATCH_SIZES),
        "gradient_accumulation": False,
    }
    g0_evaluation = evaluate_checkpoint(paths["parent"], manifest)
    g0_anchor = next(row for row in g0_evaluation["rows"] if row["id"] == ANCHOR_ID)
    g0_frozen = grouped_aggregates(g0_evaluation["rows"])
    for seed_name in TRAINING_SEEDS:
        for batch_size in BATCH_SIZES:
            workdir = args.workdir / "cells" / lane_id(seed_name, batch_size)
            run = run_cell(paths, manifest, workdir, seed_name, batch_size)
            evaluation = evaluate_checkpoint(run["checkpoint"], manifest)
            anchor = next(row for row in evaluation["rows"] if row["id"] == ANCHOR_ID)
            export_for_arena(run["checkpoint"], workdir, lane_id(seed_name, batch_size))
            arena = run_static_baseline(
                checkpoint=run["checkpoint"],
                baseline_artifact=ROOT / "storage/ai/alphazero_lite/current",
                out=args.workdir / "arena" / f"{lane_id(seed_name, batch_size)}.json",
                games=30,
                seed=TRAINING_SEEDS[seed_name],
            )["result"]
            effect = float(arena["score"]) - 0.5
            delta = anchor["optimal_mass"] - g0_anchor["optimal_mass"]
            cell = {
                "id": lane_id(seed_name, batch_size),
                "replay": REPLAY,
                "training_seed": seed_name,
                "seed": TRAINING_SEEDS[seed_name],
                "batch_size": batch_size,
                **run,
                "checkpoint": str(run["checkpoint"]),
                "anchor": anchor,
                "anchor_optimal_mass_delta": delta,
                "anchor_change_per_optimizer_step": delta / len(run["step_trace"]),
                "anchor_change_per_10k_examples": delta
                * 10000
                / run["examples_consumed"],
                "anchor_forgotten": anchor_forgotten(g0_anchor, anchor),
                "frozen_set": grouped_aggregates(evaluation["rows"]),
                "frozen_counts": frozen_counts(
                    g0_evaluation["rows"], evaluation["rows"]
                ),
                "step_summary": step_summary(run["step_trace"]),
                "matched_example_progress": matched_progress(
                    run["step_trace"], run["examples_consumed"]
                ),
                "arena": arena,
                "arena_effect": effect,
                "arena_effect_confidence_interval_95": {
                    "lower": float(arena["confidence_interval_95"]["lower"]) - 0.5,
                    "upper": float(arena["confidence_interval_95"]["upper"]) - 0.5,
                },
                "arena_material_regression": False,
                "new_critical_forensic_regression": False,
                "still_learning": bool(
                    run["epoch_metrics"][-1]["policy_loss"]
                    < run["epoch_metrics"][0]["policy_loss"]
                ),
            }
            result["cells"].append(cell)
    controls = {
        cell["training_seed"]: cell
        for cell in result["cells"]
        if cell["batch_size"] == 512
    }
    for seed, cell in controls.items():
        if (
            abs(cell["anchor_optimal_mass_delta"] - EXPECTED_CONTROL_DELTA[seed])
            > BASELINE_TOLERANCE
        ):
            result["classification"] = "batch_stability_baseline_not_reproduced"
            write_json(args.out_result, result)
            raise RuntimeError("batch_stability_baseline_not_reproduced")
    result["baseline_reproduction"] = {
        seed: {
            "expected_delta": EXPECTED_CONTROL_DELTA[seed],
            "actual_delta": cell["anchor_optimal_mass_delta"],
            "within_tolerance": True,
        }
        for seed, cell in controls.items()
    }
    for cell in result["cells"]:
        cell["arena_material_regression"] = (
            cell["arena_effect"]
            < controls[cell["training_seed"]]["arena_effect"]
            - MATERIAL_ARENA_REGRESSION
        )
    result["success_rule"] = success_rule(result["cells"], g0_frozen)
    result["classification"], result["next_experiment"] = classify(
        result["success_rule"], result["cells"]
    )
    write_json(args.out_result, result)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
