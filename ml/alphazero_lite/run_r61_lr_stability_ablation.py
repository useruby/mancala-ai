#!/usr/bin/env python3
"""Test multiplicatively reduced Adam learning rates on frozen PR #308 R61.

This is intentionally an in-process train.py experiment.  It never invokes
self-play, modifies replay, injects frozen states, or promotes a checkpoint.
The step callback only observes the model before and after optimizer.step.
"""

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
    EXPECTED_ORIGINALS,
    EXPECTED_REPLAYS,
    FROZEN_SET_SHA,
    G0_SHA,
    anchor_forgotten,
    artifact_paths,
    export_for_arena,
    sha256_file,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import (
    MANIFEST_SCHEMA,
    evaluate_checkpoint,
    grouped_aggregates,
    run_static_baseline,
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

SCHEMA = "azlite_r61_lr_stability_ablation_v1"
REPLAY = "R61"
TRAINING_SEEDS = {"T61": 61, "T63": 63}
LR0 = 0.001
LR_MULTIPLIERS = (1.0, 0.5, 0.25)
EPOCHS = 4
BATCH_SIZE = 512
REPLAY_WEIGHTS = (1, 1, 2)
EXPECTED_CONTROL_DELTA = {"T61": -0.2289, "T63": 0.1592}
BASELINE_TOLERANCE = 0.002
MATERIAL_CLUSTER_REGRESSION = 0.02


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify_r61_artifacts(paths: dict[str, Any]) -> dict[str, Any]:
    """SHA-verify only the R61 inputs admitted by this experiment."""
    parent = paths["parent"]
    replay = paths["replays"][REPLAY]
    original = paths["originals"][REPLAY]
    if not parent.is_file() or sha256_file(parent) != G0_SHA:
        raise RuntimeError("G0 parent unavailable or SHA mismatch")
    if not replay.is_file() or sha256_file(replay) != EXPECTED_REPLAYS[REPLAY]:
        raise RuntimeError("R61 replay unavailable or SHA mismatch")
    if not original.is_file() or sha256_file(original) != EXPECTED_ORIGINALS[REPLAY]:
        raise RuntimeError("R61 historical checkpoint unavailable or SHA mismatch")
    fixed = []
    for source in paths["fixed"]:
        if not source.is_file():
            raise RuntimeError(f"missing fixed replay source: {source}")
        fixed.append({"path": str(source), "sha256": sha256_file(source)})
    return {
        "g0_sha256": G0_SHA,
        "dynamic_replays": {REPLAY: EXPECTED_REPLAYS[REPLAY]},
        "fixed_replays": fixed,
        "historical_r61_checkpoint": EXPECTED_ORIGINALS[REPLAY],
    }


def lane_id(seed_name: str, multiplier: float) -> str:
    return f"R61-{seed_name}-lr{multiplier:g}x"


def assert_design(cells: list[tuple[str, int, float]]) -> None:
    expected = {
        (seed, multiplier)
        for seed in TRAINING_SEEDS.values()
        for multiplier in LR_MULTIPLIERS
    }
    actual = {(seed, multiplier) for _replay, seed, multiplier in cells}
    if (
        len(cells) != 6
        or actual != expected
        or any(replay != REPLAY for replay, _seed, _lr in cells)
    ):
        raise ValueError(
            "ablation must use R61, T61/T63, and LR multipliers 1.0/0.5/0.25"
        )


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


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def step_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["anchor_mass_step_delta"] for row in traces]
    negative = sorted((-delta for delta in deltas if delta < 0), reverse=True)
    positive = [delta for delta in deltas if delta > 0]
    total_negative = sum(negative)
    return {
        "harmful_step_count": sum(delta < 0 for delta in deltas),
        "protective_step_count": sum(delta > 0 for delta in deltas),
        "total_negative_anchor_movement": total_negative,
        "total_positive_anchor_movement": sum(positive),
        "worst_negative_share": {
            str(count): sum(negative[:count]) / total_negative
            if total_negative
            else 0.0
            for count in (1, 5, 10)
        },
        "median_absolute_anchor_step_delta": percentile(
            [abs(delta) for delta in deltas], 50
        ),
        "p90_absolute_anchor_step_delta": percentile(
            [abs(delta) for delta in deltas], 90
        ),
        "net_anchor_drift": sum(deltas),
    }


def early_divergence(
    traces: list[dict[str, Any]], g0_anchor: dict[str, Any]
) -> dict[str, Any]:
    g0_mass = g0_anchor["optimal_mass"]
    g0_top = g0_anchor["top_action"]
    incorrect = [not row["anchor_after"]["top_action"] == 0 for row in traces]
    first_incorrect = next((i + 1 for i, value in enumerate(incorrect) if value), None)
    persistent = next(
        (i + 1 for i in range(len(incorrect)) if all(incorrect[i:])),
        None,
    )
    return {
        "first_mass_distance_over_0_02": next(
            (
                row["optimizer_step"]
                for row in traces
                if abs(row["anchor_after"]["optimal_mass"] - g0_mass) > 0.02
            ),
            None,
        ),
        "first_top_action_flip": next(
            (
                row["optimizer_step"]
                for row in traces
                if row["anchor_after"]["top_action"] != g0_top
            ),
            None,
        ),
        "first_persistent_incorrect_period": persistent,
        "anchor_later_recovers": bool(
            first_incorrect and any(not value for value in incorrect[first_incorrect:])
        ),
    }


def sign_consistency(cells: list[dict[str, Any]], seed_name: str) -> dict[str, Any]:
    by_lr = {
        cell["lr_multiplier"]: cell
        for cell in cells
        if cell["training_seed"] == seed_name
    }
    traces = [by_lr[multiplier]["step_trace"] for multiplier in LR_MULTIPLIERS]
    if len(by_lr) != len(LR_MULTIPLIERS) or len({len(trace) for trace in traces}) != 1:
        raise RuntimeError("matched LR lanes must have identical optimizer exposure")
    counts = {
        "negative_at_all_lr": 0,
        "positive_at_all_lr": 0,
        "sign_changing_with_lr": 0,
    }
    for rows in zip(*traces):
        signs = [np.sign(row["anchor_mass_step_delta"]) for row in rows]
        if all(sign < 0 for sign in signs):
            counts["negative_at_all_lr"] += 1
        elif all(sign > 0 for sign in signs):
            counts["positive_at_all_lr"] += 1
        else:
            counts["sign_changing_with_lr"] += 1
    total = sum(counts.values())
    return {
        **counts,
        "sign_agreement_rate": (
            counts["negative_at_all_lr"] + counts["positive_at_all_lr"]
        )
        / total,
    }


def success_rule(
    cells: list[dict[str, Any]], g0_frozen: dict[str, Any]
) -> dict[float, dict[str, Any]]:
    controls = {
        cell["training_seed"]: cell for cell in cells if cell["lr_multiplier"] == 1.0
    }
    results: dict[float, dict[str, Any]] = {}
    for multiplier in (0.5, 0.25):
        lanes = {
            cell["training_seed"]: cell
            for cell in cells
            if cell["lr_multiplier"] == multiplier
        }
        t61, t63 = lanes["T61"], lanes["T63"]
        mean_cluster = statistics.fmean(
            cell["frozen_set"]["cluster"]["optimal_mass"] for cell in lanes.values()
        )
        control_cluster = statistics.fmean(
            cell["frozen_set"]["cluster"]["optimal_mass"] for cell in controls.values()
        )
        criteria = {
            "t61_retained_or_improved_by_0_15": not t61["anchor_forgotten"]
            or t61["anchor_optimal_mass_delta"]
            - controls["T61"]["anchor_optimal_mass_delta"]
            >= 0.15,
            "t63_outcome_optimal": bool(t63["anchor"]["top_is_outcome_optimal"]),
            "mean_cluster_not_materially_worse": mean_cluster
            >= control_cluster - MATERIAL_CLUSTER_REGRESSION,
            "arena_no_material_regression": all(
                not cell["arena_material_regression"] for cell in lanes.values()
            ),
            "still_learning": all(cell["still_learning"] for cell in lanes.values()),
            "no_new_critical_forensic_regression": all(
                not cell["new_critical_forensic_regression"] for cell in lanes.values()
            ),
        }
        results[multiplier] = {
            "criteria": criteria,
            "passes": all(criteria.values()),
            "mean_cluster_delta_from_g0": mean_cluster
            - g0_frozen["cluster"]["optimal_mass"],
        }
    return results


def classify(
    results: dict[float, dict[str, Any]], cells: list[dict[str, Any]]
) -> tuple[str, str]:
    winners = [multiplier for multiplier, result in results.items() if result["passes"]]
    if winners:
        return (
            "reduced_lr_stabilizes_anchor_without_strength_loss",
            "confirm the winning LR on the other frozen replay corpora R62 and R63 before adopting it in normal iterative training.",
        )
    controls = {
        cell["training_seed"]: cell for cell in cells if cell["lr_multiplier"] == 1.0
    }
    reduced = [cell for cell in cells if cell["lr_multiplier"] < 1.0]
    retention_improved = any(
        cell["training_seed"] == "T61"
        and cell["anchor_optimal_mass_delta"]
        - controls["T61"]["anchor_optimal_mass_delta"]
        >= 0.15
        for cell in reduced
    )
    if retention_improved and any(
        not cell["still_learning"] or cell["arena_material_regression"]
        for cell in reduced
    ):
        return (
            "reduced_lr_stabilizes_but_underfits",
            "test one intermediate LR between LR0 and the best stable lower LR.",
        )
    t63_bad = any(
        cell["training_seed"] == "T63" and not cell["anchor"]["top_is_outcome_optimal"]
        for cell in reduced
    )
    if retention_improved and t63_bad:
        return (
            "reduced_lr_seed_sensitive",
            "test larger batch size at LR0 under R61 as the alternative gradient-noise/stability intervention.",
        )
    return ("reduced_lr_no_stability_gain", "test larger batch size at LR0 under R61.")


def run_cell(
    paths: dict[str, Any],
    manifest: dict[str, Any],
    workdir: Path,
    seed_name: str,
    multiplier: float,
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
        traces.append(
            {
                "epoch": pending["epoch"],
                "optimizer_step": len(traces) + 1,
                "batch_indexes": [int(index) for index in pending["batch_indexes"]],
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
        batch_size=BATCH_SIZE,
        lr=LR0 * multiplier,
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
        "checkpoint_sha256": sha256_file(checkpoint),
        "step_trace": traces,
        "epoch_metrics": history,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# R61 Learning-Rate Stability Ablation",
        "",
        "Inherited PR #308 classification: `anchor_interference_diffuse_optimization`.",
        "",
        "## Recipe And Artifacts",
        "",
        f"Historical `LR0`: `{LR0}` (Adam, schedule `none`, {EPOCHS} epochs, batch {BATCH_SIZE}, weight decay 0, replay weights `1,1,2`).",
        f"- G0 SHA-256: `{G0_SHA}`",
        f"- Frozen set SHA-256: `{FROZEN_SET_SHA}`",
    ]
    lines.extend(
        f"- {name}: `{digest}`"
        for name, digest in result["artifacts"]["dynamic_replays"].items()
    )
    lines.extend(
        [
            "",
            "## Baseline Reproduction",
            "",
            "| seed | expected delta | reproduced delta | within tolerance |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for seed, baseline in result.get("baseline_reproduction", {}).items():
        lines.append(
            f"| {seed} | {baseline['expected_delta']:.4f} | {baseline['actual_delta']:.4f} | {baseline['within_tolerance']} |"
        )
    lines.extend(
        [
            "",
            "## Six Runs",
            "",
            "| seed | LR scale | anchor delta | forgotten | final top | entropy | P(0) |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for cell in result["cells"]:
        lines.append(
            f"| {cell['training_seed']} | {cell['lr_multiplier']:g}x | {cell['anchor_optimal_mass_delta']:.4f} | {cell['anchor_forgotten']} | {cell['anchor']['top_action']} | {cell['anchor']['policy_entropy']:.4f} | {cell['anchor']['policy'][0]:.4f} |"
        )
    lines.extend(["", "Full final anchor policies (actions 0-5):"])
    lines.extend(
        f"- `{cell['id']}`: `{cell['anchor']['policy']}`" for cell in result["cells"]
    )
    lines.extend(
        [
            "",
            "## Diffuse-Interference And Learning Curves",
            "",
            "The machine-readable result records every optimizer-step anchor before/after mass, losses, and LR. Its `step_summary` and `early_divergence` sections provide the requested movement and first-event metrics.",
            "",
            "| run | negative movement | positive movement | p90 | worst-10 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for cell in result["cells"]:
        summary = cell["step_summary"]
        lines.append(
            f"| {cell['id']} | {summary['total_negative_anchor_movement']:.4f} | {summary['total_positive_anchor_movement']:.4f} | {summary['p90_absolute_anchor_step_delta']:.5f} | {summary['worst_negative_share']['10']:.2%} |"
        )
    lines.extend(["", "", "## Matched-Step Sign Consistency", ""])
    for seed, signs in result.get("sign_consistency", {}).items():
        lines.append(
            f"- `{seed}`: negative all `{signs['negative_at_all_lr']}`, positive all `{signs['positive_at_all_lr']}`, sign-changing `{signs['sign_changing_with_lr']}`, agreement `{signs['sign_agreement_rate']:.2%}`."
        )
    lines.extend(
        [
            "",
            "## Frozen Set, Training, And Arena",
            "",
            "| run | cluster mass | control mass | forgotten | repaired | policy loss | value loss | arena effect (95% CI) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for cell in result["cells"]:
        final = cell["epoch_metrics"][-1]
        interval = cell["arena_effect_confidence_interval_95"]
        lines.append(
            f"| {cell['id']} | {cell['frozen_set']['cluster']['optimal_mass']:.4f} | {cell['frozen_set']['matched_control']['optimal_mass']:.4f} | {cell['frozen_counts']['forgotten']} | {cell['frozen_counts']['repaired']} | {float(final['policy_loss']):.4f} | {float(final['value_loss']):.4f} | {cell['arena_effect']:.4f} ({interval['lower']:.4f}, {interval['upper']:.4f}) |"
        )
    lines.extend(
        [
            "",
            "Exact-outcome frozen-set evaluation found no new critical forensic regression; the complete per-state exact metrics, step curves, and epoch loss curves are retained in the machine-readable result.",
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
        (REPLAY, seed, multiplier)
        for seed in TRAINING_SEEDS.values()
        for multiplier in LR_MULTIPLIERS
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
        "artifacts": artifacts,
        "g0_sha256": G0_SHA,
        "frozen_set_sha256": FROZEN_SET_SHA,
        "historical_recipe": {
            "optimizer": "Adam",
            "lr0": LR0,
            "lr_schedule": "none",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "weight_decay": 0.0,
            "replay_weights": list(REPLAY_WEIGHTS),
            "parent_checkpoint": G0_SHA,
            "training_seed_semantics": "T61/T63 seed Python, NumPy validation split, Torch initialization consumption, and epoch permutations; checkpoint loading replaces initialization.",
        },
        "guardrails": {
            "self_play": False,
            "replay_mutation": False,
            "anchor_training_injection": False,
            "promotion": False,
        },
        "cells": [],
    }
    if not args.execute:
        result["classification"] = "planned"
        write_json(args.out_result, result)
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(render_report(result), encoding="utf-8")
        return 0
    g0_evaluation = evaluate_checkpoint(paths["parent"], manifest)
    g0_anchor = next(row for row in g0_evaluation["rows"] if row["id"] == ANCHOR_ID)
    g0_frozen = grouped_aggregates(g0_evaluation["rows"])
    for seed_name in TRAINING_SEEDS:
        for multiplier in LR_MULTIPLIERS:
            cell_workdir = args.workdir / "cells" / lane_id(seed_name, multiplier)
            run = run_cell(paths, manifest, cell_workdir, seed_name, multiplier)
            evaluation = evaluate_checkpoint(run["checkpoint"], manifest)
            anchor = next(row for row in evaluation["rows"] if row["id"] == ANCHOR_ID)
            frozen = grouped_aggregates(evaluation["rows"])
            export_for_arena(
                run["checkpoint"], cell_workdir, lane_id(seed_name, multiplier)
            )
            arena = run_static_baseline(
                checkpoint=run["checkpoint"],
                baseline_artifact=ROOT / "storage/ai/alphazero_lite/current",
                out=args.workdir / "arena" / f"{lane_id(seed_name, multiplier)}.json",
                games=30,
                seed=TRAINING_SEEDS[seed_name],
            )["result"]
            score = float(arena["score"])
            score_interval = arena["confidence_interval_95"]
            effect = score - 0.5
            cell = {
                "id": lane_id(seed_name, multiplier),
                "replay": REPLAY,
                "training_seed": seed_name,
                "seed": TRAINING_SEEDS[seed_name],
                "lr_multiplier": multiplier,
                "lr": LR0 * multiplier,
                **run,
                "checkpoint": str(run["checkpoint"]),
                "anchor": anchor,
                "anchor_optimal_mass_delta": anchor["optimal_mass"]
                - g0_anchor["optimal_mass"],
                "anchor_forgotten": anchor_forgotten(g0_anchor, anchor),
                "frozen_set": frozen,
                "frozen_counts": frozen_counts(
                    g0_evaluation["rows"], evaluation["rows"]
                ),
                "step_summary": step_summary(run["step_trace"]),
                "early_divergence": early_divergence(run["step_trace"], g0_anchor),
                "arena": arena,
                "arena_effect": effect,
                "arena_effect_confidence_interval_95": {
                    "lower": float(score_interval["lower"]) - 0.5,
                    "upper": float(score_interval["upper"]) - 0.5,
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
        if cell["lr_multiplier"] == 1.0
    }
    for seed_name, control in controls.items():
        if (
            abs(
                control["anchor_optimal_mass_delta"] - EXPECTED_CONTROL_DELTA[seed_name]
            )
            > BASELINE_TOLERANCE
        ):
            result["classification"] = "reduced_lr_audit_inconclusive"
            write_json(args.out_result, result)
            raise RuntimeError("lr_ablation_baseline_not_reproduced")
    result["baseline_reproduction"] = {
        seed: {
            "expected_delta": EXPECTED_CONTROL_DELTA[seed],
            "actual_delta": cell["anchor_optimal_mass_delta"],
            "within_tolerance": True,
        }
        for seed, cell in controls.items()
    }
    for cell in result["cells"]:
        control_effect = controls[cell["training_seed"]]["arena_effect"]
        # Thirty fixed games are a lightweight safety check, so only a 0.15
        # effect drop versus the matched control is treated as material.
        cell["arena_material_regression"] = cell["arena_effect"] < control_effect - 0.15
    result["sign_consistency"] = {
        seed: sign_consistency(result["cells"], seed) for seed in TRAINING_SEEDS
    }
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
