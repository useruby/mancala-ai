#!/usr/bin/env python3
"""Gradient-clip-strength-only audit on frozen R61 replay.

This runner is observational apart from the configured global clip threshold.
It never generates self-play, mutates replay, injects anchors, or promotes a
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
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

SCHEMA = "azlite_r61_grad_clip_stability_ablation_v1"
TRAINING_SEEDS = {"T61": 61, "T63": 63}
CLIP_LANES: tuple[tuple[str, float | None], ...] = (
    ("C1", 1.0),
    ("C2", 2.0),
    ("C4", 4.0),
    ("CNONE", None),
)
BATCH_SIZE = 512
LR0 = 0.001
EPOCHS = 4
ANCHOR_IMPROVEMENT_THRESHOLD = 0.15
MATERIAL_CLUSTER_REGRESSION = 0.02
MATERIAL_ARENA_REGRESSION = 0.15
MATERIAL_STEP_DIFFERENCE = 0.05
GRADIENT_NORM_SAFETY_THRESHOLD = 100.0
LOSS_SPIKE_THRESHOLD = 10.0


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def lane_id(seed_name: str, lane: str) -> str:
    return f"R61-{seed_name}-{lane.lower()}"


def assert_design(cells: list[tuple[str, int, int, float, float | None]]) -> None:
    expected = {
        (seed, clip) for seed in TRAINING_SEEDS.values() for _, clip in CLIP_LANES
    }
    actual = {(seed, clip) for replay, seed, _batch, _lr, clip in cells}
    if (
        len(cells) != 8
        or actual != expected
        or any(
            replay != REPLAY or batch != BATCH_SIZE or lr != LR0
            for replay, _seed, batch, lr, _clip in cells
        )
    ):
        raise ValueError(
            "ablation must use only R61, T61/T63, B512, LR0, four epochs, and C1/C2/C4/CNONE"
        )


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def step_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["anchor_mass_step_delta"]) for row in traces]
    absolute = [abs(delta) for delta in deltas]
    postclip = [float(row["post_clip_gradient_norm"]) for row in traces]
    preclip = [float(row["preclip_gradient_norm"]) for row in traces]
    scales = [float(row["clip_scale"]) for row in traces]
    return {
        "optimizer_steps": len(traces),
        "harmful_step_count": sum(delta < 0 for delta in deltas),
        "protective_step_count": sum(delta > 0 for delta in deltas),
        "total_negative_anchor_movement": sum(-delta for delta in deltas if delta < 0),
        "total_positive_anchor_movement": sum(delta for delta in deltas if delta > 0),
        "median_absolute_anchor_step_delta": percentile(absolute, 50),
        "p90_absolute_anchor_step_delta": percentile(absolute, 90),
        "anchor_step_delta_variance": statistics.variance(deltas)
        if len(deltas) > 1
        else 0.0,
        "anchor_delta_per_unit_postclip_grad_norm": sum(deltas) / sum(postclip)
        if sum(postclip)
        else 0.0,
        "clipping_exposure": {
            "fraction_steps_clipped": statistics.fmean(
                bool(row["clip_active"]) for row in traces
            )
            if traces
            else 0.0,
            "preclip_norm": {f"p{q}": percentile(preclip, q) for q in (50, 90, 95, 99)},
            "clip_scale": {f"p{q}": percentile(scales, q) for q in (50, 90)},
            "mean_postclip_norm": statistics.fmean(postclip) if postclip else 0.0,
            "total_unclipped_gradient_norm_exposure": sum(preclip),
            "total_postclip_gradient_norm_exposure": sum(postclip),
        },
    }


def early_divergence(traces: list[dict[str, Any]]) -> dict[str, Any]:
    incorrect = [row["anchor_after"]["top_action"] != 0 for row in traces]
    longest = current = 0
    for value in incorrect:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "first_top_action_flip": next(
            (
                row["optimizer_step"]
                for row in traces
                if row["anchor_after"]["top_action"] != 0
            ),
            None,
        ),
        "first_mass_below_0_20": next(
            (
                row["optimizer_step"]
                for row in traces
                if row["anchor_after"]["optimal_mass"] < 0.20
            ),
            None,
        ),
        "first_mass_above_0_50": next(
            (
                row["optimizer_step"]
                for row in traces
                if row["anchor_after"]["optimal_mass"] > 0.50
            ),
            None,
        ),
        "longest_continuous_incorrect_interval": longest,
        "final_correctness": not incorrect[-1] if incorrect else None,
    }


def clipped_step_analysis(traces: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "steps": len(rows),
            "mean_anchor_step_delta": statistics.fmean(
                float(row["anchor_mass_step_delta"]) for row in rows
            )
            if rows
            else 0.0,
            "harmful_step_rate": statistics.fmean(
                float(row["anchor_mass_step_delta"] < 0) for row in rows
            )
            if rows
            else 0.0,
            "protective_step_rate": statistics.fmean(
                float(row["anchor_mass_step_delta"] > 0) for row in rows
            )
            if rows
            else 0.0,
            "policy_loss": statistics.fmean(float(row["policy_loss"]) for row in rows)
            if rows
            else 0.0,
            "value_loss": statistics.fmean(float(row["value_loss"]) for row in rows)
            if rows
            else 0.0,
        }

    return {
        "clipped": summarize([row for row in traces if row["clip_active"]]),
        "unclipped": summarize([row for row in traces if not row["clip_active"]]),
    }


def optimizer_summary(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    previous: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "shared_trunk": [],
        "policy_head": [],
        "value_head": [],
    }
    for name, parameter in model.named_parameters():
        group = (
            "policy_head"
            if name.startswith("policy_")
            else "value_head"
            if name.startswith("value_")
            else "shared_trunk"
        )
        groups[group].append((name, parameter))
    result = {}
    for group, named_parameters in groups.items():
        moments = [
            optimizer.state[p] for _, p in named_parameters if p in optimizer.state
        ]
        exp_avg = [
            state["exp_avg"].detach().reshape(-1)
            for state in moments
            if "exp_avg" in state
        ]
        exp_avg_sq = [
            state["exp_avg_sq"].detach().reshape(-1)
            for state in moments
            if "exp_avg_sq" in state
        ]
        updates = [
            (p.detach() - previous[name]).reshape(-1) for name, p in named_parameters
        ]
        result[group] = {
            "mean_absolute_first_moment": float(torch.cat(exp_avg).abs().mean())
            if exp_avg
            else 0.0,
            "second_moment_norm": float(torch.linalg.vector_norm(torch.cat(exp_avg_sq)))
            if exp_avg_sq
            else 0.0,
            "parameter_update_norm": float(torch.linalg.vector_norm(torch.cat(updates)))
            if updates
            else 0.0,
        }
    return result


def run_cell(
    paths: dict[str, Any],
    manifest: dict[str, Any],
    workdir: Path,
    seed_name: str,
    lane: str,
    grad_clip: float | None,
) -> dict[str, Any]:
    set_seed(TRAINING_SEEDS[seed_name])
    x, p, v, replay_indexes = load_jsonl_replay(
        [paths["replays"][REPLAY], *paths["fixed"]],
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
    adam_epochs: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    previous = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

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
                "batch_indexes": [int(i) for i in pending["batch_indexes"]],
                "lr": pending["lr"],
                "policy_loss": pending["policy_loss"],
                "value_loss": pending["value_loss"],
                "preclip_gradient_norm": pending["gradient_norm"],
                "clip_threshold": pending["grad_clip"],
                "clip_active": pending["clip_active"],
                "clip_scale": pending["clip_scale"],
                "post_clip_gradient_norm": pending["post_clip_gradient_norm"],
                "anchor_before": pending["anchor_before"],
                "anchor_after": after,
                "anchor_mass_step_delta": after["optimal_mass"]
                - pending["anchor_before"]["optimal_mass"],
            }
        )
        pending = None

    def epoch_callback(
        epoch: int, optimizer: torch.optim.Optimizer, current: torch.nn.Module
    ) -> None:
        nonlocal previous
        adam_epochs.append(
            {"epoch": epoch, "groups": optimizer_summary(optimizer, current, previous)}
        )
        previous = {
            name: parameter.detach().clone()
            for name, parameter in current.named_parameters()
        }

    history: list[dict[str, float | int | None]] = []
    train(
        model,
        x,
        p,
        v,
        replay_indexes,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR0,
        device=torch.device("cpu"),
        value_loss_weight=0.3,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.1,
        grad_clip=grad_clip,
        save_top_k=3,
        lr_scheduler="none",
        epoch_history=history,
        step_callback=callback,
        epoch_callback=epoch_callback,
    )
    checkpoint = workdir / "checkpoint.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint, **checkpoint_from_model(model))
    return {
        "checkpoint": checkpoint,
        "step_trace": traces,
        "epoch_metrics": history,
        "adam_state_by_epoch": adam_epochs,
    }


def frozen_counts(
    g0_rows: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, int]:
    g0 = {row["id"]: row for row in g0_rows}
    cluster = [row for row in rows if row["membership"] != "matched_control"]
    return {
        "forgotten": sum(
            g0[row["id"]]["top_is_outcome_optimal"]
            and not row["top_is_outcome_optimal"]
            for row in cluster
        ),
        "repaired": sum(
            not g0[row["id"]]["top_is_outcome_optimal"]
            and row["top_is_outcome_optimal"]
            for row in cluster
        ),
    }


def success_rule(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    controls = {
        cell["training_seed"]: cell for cell in cells if cell["clip_lane"] == "C1"
    }
    result = {}
    for lane, _clip in CLIP_LANES[1:]:
        lanes = {
            cell["training_seed"]: cell for cell in cells if cell["clip_lane"] == lane
        }
        mean_cluster = statistics.fmean(
            cell["frozen_set"]["cluster"]["optimal_mass"] for cell in lanes.values()
        )
        control_cluster = statistics.fmean(
            cell["frozen_set"]["cluster"]["optimal_mass"] for cell in controls.values()
        )
        criteria = {
            "t61_improves_by_0_15_or_is_optimal": lanes["T61"][
                "anchor_optimal_mass_delta"
            ]
            - controls["T61"]["anchor_optimal_mass_delta"]
            >= ANCHOR_IMPROVEMENT_THRESHOLD
            or lanes["T61"]["anchor"]["top_is_outcome_optimal"],
            "t63_outcome_optimal": lanes["T63"]["anchor"]["top_is_outcome_optimal"],
            "mean_cluster_not_materially_worse": mean_cluster
            >= control_cluster - MATERIAL_CLUSTER_REGRESSION,
            "arena_no_material_regression": all(
                not cell["arena_material_regression"] for cell in lanes.values()
            ),
            "no_new_critical_forensic_regression": all(
                not cell["new_critical_forensic_regression"] for cell in lanes.values()
            ),
            "numerically_stable": all(
                not cell["numerical_instability"] for cell in lanes.values()
            ),
            "clipping_exposure_changed": all(
                abs(
                    cell["step_summary"]["clipping_exposure"]["fraction_steps_clipped"]
                    - controls[seed]["step_summary"]["clipping_exposure"][
                        "fraction_steps_clipped"
                    ]
                )
                > 0.05
                or abs(
                    cell["step_summary"]["clipping_exposure"]["clip_scale"]["p50"]
                    - controls[seed]["step_summary"]["clipping_exposure"]["clip_scale"][
                        "p50"
                    ]
                )
                > 0.05
                for seed, cell in lanes.items()
            ),
        }
        result[lane] = {
            "criteria": criteria,
            "passes": all(criteria.values()),
            "mean_cluster": mean_cluster,
            "mean_cluster_delta_from_c1": mean_cluster - control_cluster,
        }
    return result


def classify(
    results: dict[str, dict[str, Any]], cells: list[dict[str, Any]]
) -> tuple[str, str]:
    winners = [lane for lane in ("C2", "C4", "CNONE") if results[lane]["passes"]]
    if winners:
        winner = winners[0]
        if winner in {"C2", "C4"}:
            return (
                "relaxed_grad_clip_stabilizes_anchor",
                f"confirm grad_clip={dict(CLIP_LANES)[winner]} on frozen R62 and R63 with T61/T63 before changing normal training.",
            )
        return (
            "no_grad_clip_stabilizes_anchor",
            "confirm no-clipping versus C1 on R62/R63 before considering a production change.",
        )
    controls = {
        cell["training_seed"]: cell for cell in cells if cell["clip_lane"] == "C1"
    }
    relaxed = [cell for cell in cells if cell["clip_lane"] != "C1"]
    if any(
        cell["numerical_instability"] or cell["arena_material_regression"]
        for cell in relaxed
    ):
        return (
            "grad_clip_relaxation_unstable",
            "keep clip=1.0 and audit Adam moment/parameter-subspace drift; do not tighten clipping further.",
        )
    improves = any(
        cell["training_seed"] == "T61"
        and cell["anchor_optimal_mass_delta"]
        - controls["T61"]["anchor_optimal_mass_delta"]
        >= ANCHOR_IMPROVEMENT_THRESHOLD
        for cell in relaxed
    )
    loses_t63 = any(
        cell["training_seed"] == "T63" and not cell["anchor"]["top_is_outcome_optimal"]
        for cell in relaxed
    )
    if improves and loses_t63:
        return (
            "grad_clip_relaxation_seed_sensitive",
            "stop scalar optimizer tuning and audit Adam moment/parameter-subspace drift between the successful and failing T61/T63 trajectories.",
        )
    return (
        "grad_clip_strength_no_stability_gain",
        "audit which parameter subspaces (shared trunk vs policy head vs value head) carry the anchor-damaging drift under the already-frozen R61/T61 vs R61/T63 trajectories.",
    )


def matched_steps(cells: list[dict[str, Any]], seed: str) -> dict[str, Any]:
    by_lane = {
        cell["clip_lane"]: cell["step_trace"]
        for cell in cells
        if cell["training_seed"] == seed
    }
    traces = [by_lane[lane] for lane, _ in CLIP_LANES]
    if len(by_lane) != 4 or len({len(trace) for trace in traces}) != 1:
        raise RuntimeError("matched clip lanes must have identical optimizer exposure")
    rows = []
    for lane_rows in zip(*traces):
        batch_hashes = {
            hashlib.sha256(json.dumps(row["batch_indexes"]).encode()).hexdigest()
            for row in lane_rows
        }
        if len(batch_hashes) != 1:
            raise RuntimeError("clip threshold changed minibatch order")
        deltas = {
            lane: float(row["anchor_mass_step_delta"])
            for (lane, _), row in zip(CLIP_LANES, lane_rows)
        }
        masses = {
            lane: float(row["anchor_after"]["optimal_mass"])
            for (lane, _), row in zip(CLIP_LANES, lane_rows)
        }
        sensitive = (
            max(deltas.values()) - min(deltas.values()) >= MATERIAL_STEP_DIFFERENCE
        )
        if sensitive:
            rows.append(
                {
                    "optimizer_step": lane_rows[0]["optimizer_step"],
                    "batch_hash": batch_hashes.pop(),
                    "anchor_step_deltas": deltas,
                    "anchor_optimal_mass_after": masses,
                    "signs": {
                        lane: int(np.sign(value)) for lane, value in deltas.items()
                    },
                }
            )
    return {
        "matched_minibatch_order": True,
        "clip_sensitive_step_threshold": MATERIAL_STEP_DIFFERENCE,
        "clip_sensitive_step_count": len(rows),
        "clip_sensitive_steps": rows,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# R61 Gradient Clip Strength Ablation",
        "",
        "Inherited PR #310 classification: `larger_batch_no_stability_gain`.",
        "",
        "Historical B512/LR0 already used global `grad_clip=1.0`; this is threshold relaxation, not clipping off -> on.",
        "",
        "## Artifacts",
        "",
        f"- G0 SHA-256: `{G0_SHA}`",
        f"- R61 replay SHA-256: `{result['artifacts']['dynamic_replays'][REPLAY]}`",
        f"- Frozen set SHA-256: `{FROZEN_SET_SHA}`",
        "- Adam, LR `0.001`, B512, four epochs, schedule `none`, and replay weights `1,1,2` are fixed.",
        f"- G0 anchor optimal mass: `{result.get('g0_anchor_optimal_mass', 0.0):.4f}`.",
        "",
        "## C1 Baseline Reproduction",
        "",
        "| seed | expected delta | observed | within tolerance |",
        "| --- | ---: | ---: | --- |",
    ]
    for seed, row in result.get("baseline_reproduction", {}).items():
        lines.append(
            f"| {seed} | {row['expected_delta']:.4f} | {row['actual_delta']:.4f} | {row['within_tolerance']} |"
        )
    lines.extend(
        [
            "",
            "## Eight Runs And Clip Exposure",
            "",
            "| run | clip | anchor delta | forgotten | top | P(0) | entropy | clipped | p50/p90 preclip | p50/p90 scale |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for cell in result.get("cells", []):
        exposure = cell["step_summary"]["clipping_exposure"]
        lines.append(
            f"| {cell['id']} | {cell['grad_clip']} | {cell['anchor_optimal_mass_delta']:.4f} | {cell['anchor_forgotten']} | {cell['anchor']['top_action']} | {cell['anchor']['policy'][0]:.4f} | {cell['anchor']['policy_entropy']:.4f} | {exposure['fraction_steps_clipped']:.1%} | {exposure['preclip_norm']['p50']:.3f}/{exposure['preclip_norm']['p90']:.3f} | {exposure['clip_scale']['p50']:.3f}/{exposure['clip_scale']['p90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Full Clipping Exposure",
            "",
            "| run | p50/p90/p95/p99 preclip | p50/p90 scale | mean postclip | total preclip | total postclip |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for cell in result.get("cells", []):
        exposure = cell["step_summary"]["clipping_exposure"]
        preclip, scale = exposure["preclip_norm"], exposure["clip_scale"]
        lines.append(
            f"| {cell['id']} | {preclip['p50']:.3f}/{preclip['p90']:.3f}/{preclip['p95']:.3f}/{preclip['p99']:.3f} | {scale['p50']:.3f}/{scale['p90']:.3f} | {exposure['mean_postclip_norm']:.3f} | {exposure['total_unclipped_gradient_norm_exposure']:.1f} | {exposure['total_postclip_gradient_norm_exposure']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Step Dynamics",
            "",
            "| run | harmful/protective | negative/positive movement | median/p90 abs delta | variance | delta/postclip norm | first flip | <0.20 | >0.50 | longest incorrect |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cell in result.get("cells", []):
        summary, early = cell["step_summary"], cell["early_divergence"]
        lines.append(
            f"| {cell['id']} | {summary['harmful_step_count']}/{summary['protective_step_count']} | {summary['total_negative_anchor_movement']:.3f}/{summary['total_positive_anchor_movement']:.3f} | {summary['median_absolute_anchor_step_delta']:.4f}/{summary['p90_absolute_anchor_step_delta']:.4f} | {summary['anchor_step_delta_variance']:.6f} | {summary['anchor_delta_per_unit_postclip_grad_norm']:.6f} | {early['first_top_action_flip']} | {early['first_mass_below_0_20']} | {early['first_mass_above_0_50']} | {early['longest_continuous_incorrect_interval']} |"
        )
    lines.extend(
        [
            "",
            "## Clipped Versus Unclipped (Descriptive)",
            "",
            "| run | group | mean delta | harmful rate | protective rate | policy loss | value loss |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cell in result.get("cells", []):
        for group, values in cell["clipped_vs_unclipped"].items():
            lines.append(
                f"| {cell['id']} | {group} ({values['steps']}) | {values['mean_anchor_step_delta']:.5f} | {values['harmful_step_rate']:.1%} | {values['protective_step_rate']:.1%} | {values['policy_loss']:.4f} | {values['value_loss']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Matched Steps",
            "",
            f"Same-seed minibatch order was verified across all lanes. A clip-sensitive step is pre-registered as absolute anchor-step-delta range >= `{MATERIAL_STEP_DIFFERENCE}`. Full matched trajectories and future masses are in the JSON result; clipped-versus-unclipped associations are descriptive only.",
        ]
    )
    for seed, match in result.get("matched_steps", {}).items():
        lines.append(
            f"- `{seed}`: `{match['clip_sensitive_step_count']}` clip-sensitive steps."
        )
    lines.extend(
        [
            "",
            "## Adam State By Epoch",
            "",
            "The JSON records compact first-moment, second-moment, and parameter-update norms for shared trunk, policy head, and value head after every epoch. The final policy-head/trunk values are:",
            "",
            "| run | trunk m1/m2/update | policy m1/m2/update |",
            "| --- | --- | --- |",
        ]
    )
    for cell in result.get("cells", []):
        groups = cell["adam_state_by_epoch"][-1]["groups"]
        trunk, policy = groups["shared_trunk"], groups["policy_head"]
        lines.append(
            f"| {cell['id']} | {trunk['mean_absolute_first_moment']:.5f}/{trunk['second_moment_norm']:.5f}/{trunk['parameter_update_norm']:.5f} | {policy['mean_absolute_first_moment']:.5f}/{policy['second_moment_norm']:.5f}/{policy['parameter_update_norm']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Set, Convergence, And Safety",
            "",
            "| run | cluster mass | control mass | cluster-control | forgotten | repaired | policy/value loss | validation policy/value | arena effect (95% CI) | stable |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for cell in result.get("cells", []):
        frozen, final, interval = (
            cell["frozen_set"],
            cell["epoch_metrics"][-1],
            cell["arena_effect_confidence_interval_95"],
        )
        lines.append(
            f"| {cell['id']} | {frozen['cluster']['optimal_mass']:.4f} | {frozen['matched_control']['optimal_mass']:.4f} | {frozen['cluster']['optimal_mass'] - frozen['matched_control']['optimal_mass']:.4f} | {cell['frozen_counts']['forgotten']} | {cell['frozen_counts']['repaired']} | {float(final['policy_loss']):.4f}/{float(final['value_loss']):.4f} | {float(final.get('validation_policy_loss', 0.0)):.4f}/{float(final.get('validation_value_loss', 0.0)):.4f} | {cell['arena_effect']:.4f} ({interval['lower']:.4f}, {interval['upper']:.4f}) | {not cell['numerical_instability']} |"
        )
    lines.extend(
        [
            "",
            f"The unchanged frozen exact-outcome set is the forensic safety evaluation; no new critical correct-to-incorrect state relative to C1 is permitted. Safety limits were preclip norm <= `{GRADIENT_NORM_SAFETY_THRESHOLD:g}` and loss <= `{LOSS_SPIKE_THRESHOLD:g}`; every lane records NaN/infinity and spike status in JSON. Per-step norms, losses, anchor trajectories, clipped/unclipped summaries, and epoch Adam summaries are retained in the machine-readable artifact. No self-play, replay mutation, or promotion occurred.",
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
        (REPLAY, seed, BATCH_SIZE, LR0, clip)
        for seed in TRAINING_SEEDS.values()
        for _, clip in CLIP_LANES
    ]
    assert_design(design)
    paths, artifacts = artifact_paths(), verify_r61_artifacts(artifact_paths())
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
        "inherited_classification": "larger_batch_no_stability_gain",
        "artifacts": artifacts,
        "historical_recipe": {
            "optimizer": "Adam",
            "lr": LR0,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "grad_clip": 1.0,
            "replay_weights": list(REPLAY_WEIGHTS),
            "lr_schedule": "none",
        },
        "guardrails": {
            "self_play": False,
            "replay_mutation": False,
            "anchor_training_injection": False,
            "promotion": False,
        },
        "safety_thresholds": {
            "preclip_gradient_norm": GRADIENT_NORM_SAFETY_THRESHOLD,
            "loss": LOSS_SPIKE_THRESHOLD,
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
    result["g0_anchor_optimal_mass"] = g0_anchor["optimal_mass"]
    for seed in TRAINING_SEEDS:
        for lane, clip in CLIP_LANES:
            workdir = args.workdir / "cells" / lane_id(seed, lane)
            run = run_cell(paths, manifest, workdir, seed, lane, clip)
            evaluation = evaluate_checkpoint(run["checkpoint"], manifest)
            anchor = next(row for row in evaluation["rows"] if row["id"] == ANCHOR_ID)
            export_for_arena(run["checkpoint"], workdir, lane_id(seed, lane))
            arena = run_static_baseline(
                checkpoint=run["checkpoint"],
                baseline_artifact=ROOT / "storage/ai/alphazero_lite/current",
                out=args.workdir / "arena" / f"{lane_id(seed, lane)}.json",
                games=30,
                seed=TRAINING_SEEDS[seed],
            )["result"]
            traces = run["step_trace"]
            finite = all(
                np.isfinite(
                    [
                        row["policy_loss"],
                        row["value_loss"],
                        row["preclip_gradient_norm"],
                    ]
                ).all()
                for row in traces
            )
            numerical_instability = not finite or any(
                row["preclip_gradient_norm"] > GRADIENT_NORM_SAFETY_THRESHOLD
                or row["policy_loss"] > LOSS_SPIKE_THRESHOLD
                or row["value_loss"] > LOSS_SPIKE_THRESHOLD
                for row in traces
            )
            result["cells"].append(
                {
                    "id": lane_id(seed, lane),
                    "replay": REPLAY,
                    "training_seed": seed,
                    "seed": TRAINING_SEEDS[seed],
                    "clip_lane": lane,
                    "grad_clip": clip,
                    **run,
                    "checkpoint": str(run["checkpoint"]),
                    "anchor": anchor,
                    "g0_anchor_optimal_mass": g0_anchor["optimal_mass"],
                    "anchor_optimal_mass_delta": anchor["optimal_mass"]
                    - g0_anchor["optimal_mass"],
                    "anchor_forgotten": anchor_forgotten(g0_anchor, anchor),
                    "frozen_set": grouped_aggregates(evaluation["rows"]),
                    "frozen_counts": frozen_counts(
                        g0_evaluation["rows"], evaluation["rows"]
                    ),
                    "step_summary": step_summary(traces),
                    "clipped_vs_unclipped": clipped_step_analysis(traces),
                    "early_divergence": early_divergence(traces),
                    "arena": arena,
                    "arena_effect": float(arena["score"]) - 0.5,
                    "arena_effect_confidence_interval_95": {
                        "lower": float(arena["confidence_interval_95"]["lower"]) - 0.5,
                        "upper": float(arena["confidence_interval_95"]["upper"]) - 0.5,
                    },
                    "new_critical_forensic_regression": False,
                    "numerical_instability": numerical_instability,
                    "nan_or_infinity": not finite,
                }
            )
    controls = {
        cell["training_seed"]: cell
        for cell in result["cells"]
        if cell["clip_lane"] == "C1"
    }
    for seed, cell in controls.items():
        if (
            abs(cell["anchor_optimal_mass_delta"] - EXPECTED_CONTROL_DELTA[seed])
            > BASELINE_TOLERANCE
        ):
            result["classification"] = "grad_clip_baseline_not_reproduced"
            write_json(args.out_result, result)
            raise RuntimeError("grad_clip_baseline_not_reproduced")
    result["baseline_reproduction"] = {
        seed: {
            "expected_delta": EXPECTED_CONTROL_DELTA[seed],
            "actual_delta": cell["anchor_optimal_mass_delta"],
            "within_tolerance": True,
        }
        for seed, cell in controls.items()
    }
    for cell in result["cells"]:
        control = controls[cell["training_seed"]]
        cell["arena_material_regression"] = (
            cell["arena_effect"] < control["arena_effect"] - MATERIAL_ARENA_REGRESSION
        )
        cell["new_critical_forensic_regression"] = (
            cell["frozen_counts"]["forgotten"] > control["frozen_counts"]["forgotten"]
        )
    result["matched_steps"] = {
        seed: matched_steps(result["cells"], seed) for seed in TRAINING_SEEDS
    }
    result["success_rule"] = success_rule(result["cells"])
    result["classification"], result["next_experiment"] = classify(
        result["success_rule"], result["cells"]
    )
    write_json(args.out_result, result)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
