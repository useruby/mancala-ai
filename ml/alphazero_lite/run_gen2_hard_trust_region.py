#!/usr/bin/env python3
# ruff: noqa: E402
"""Parent-relative hard output-space trust region experiment for Gen-2 AlphaZero update.

Tests whether enforcing a strict cumulative output-space L1 divergence constraint
relative to parent P1 prevents the Gen-2 search cliff observed in PR #204.

Lineage:
- P0: original frozen incumbent from model-artifact/current
- P1: exact PR #203 beta_095 step46 parent
- P2 candidates: trained on PR #204 Gen-2 replay with beta=0.95 same-state anchoring to P1,
  under hard output-space trust-region projection.

Lanes:
- unprojected: exact PR #204 reproduction (no output-space constraint)
- l1_0010: mean legal L1 vs P1 <= 0.00100
- l1_00125: mean legal L1 vs P1 <= 0.00125 (PRIMARY)
- l1_0015: mean legal L1 vs P1 <= 0.00150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.evaluation_metrics import (  # noqa: E402
    paired_opening_candidate_effect,
)
from ml.alphazero_lite.hard_projection import (  # noqa: E402
    POLICY_KEYS,
    TrustStateSet,
    project_policy_head_step,
)
from ml.alphazero_lite.pipeline import (  # noqa: E402
    materialize_weights_json_checkpoint,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    build_manifest,
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (  # noqa: E402
    CHECKPOINT_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LR,
    PROBE_SIZE,
    PUCT_CONTEXT,
    STAGE1_CONTEXT,
    STAGE2_CONTEXT,
    TRAINABLE_SCOPE,
    _batch,
    _cross_entropy,
    _losses,
    _save_snapshot,
    _win_draw_loss,
    arena_safe,
    assert_legal_distribution,
    export_snapshot_artifacts,
    group_delta,
    group_parameters_identical,
    incumbent_policy_batch,
    mixed_policy_target,
    policy_drift_metrics,
    puct_probe,
    puct_trajectory,
    tensors_identical,
    trunk_parameters_identical,
)
from ml.alphazero_lite.run_frozen_trunk_head_isolation_ablation import (  # noqa: E402
    VALUE_STACK_PREFIXES,
)
from ml.alphazero_lite.run_gen2_selfplay_anchor_iteration import (  # noqa: E402
    CURRENT_HASH,
    GEN2_DEFAULT_SEED,
    reconstruct_and_freeze_p1,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    _new_model,
)
from ml.alphazero_lite.run_policy_detached_trunk_ablation import _arena_records  # noqa: E402
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    decoded_validation_manifest,
    model_outputs,
    stable_hash,
)
from ml.alphazero_lite.train import (  # noqa: E402
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

NAMESPACE = "azlite_hard_parent_trust_region_v1"
PR204_GEN2_REPLAY_HASH = (
    "2cee30547f8bc5d7cad6f02f859ee5e8644386e9b59c8a054ef74548c72ce84b"
)
PR204_UNPROJECTED_S46_STATE_HASH = (
    "336496d5fb33331240178c4b834b8faf9548e3915b45c9b5f7e4b7aad6626870"
)

# Radius configurations for hard projection lanes
RADIUS_LANES: list[tuple[str, float | None]] = [
    ("unprojected", None),
    ("l1_0010", 0.00100),
    ("l1_00125", 0.00125),
    ("l1_0015", 0.00150),
]

CLASSIFICATIONS = [
    "hard_projection_rescues_gen2",
    "projection_safe_but_learning_collapsed",
    "mean_l1_radius_not_sufficient",
    "radius_boundary_confirmed",
    "unprojected_reproduction_failure",
    "projection_invariant_failure",
    "inconclusive",
]

NEXT_EXPERIMENTS = {
    "hard_projection_rescues_gen2": (
        "proceed to a 3-generation lineage rollout using parent-relative "
        "hard output-space projection at each generation"
    ),
    "projection_safe_but_learning_collapsed": (
        "investigate adaptive step-size scaling or alternative projection "
        "geometries to restore supervised target progress inside the safe zone"
    ),
    "mean_l1_radius_not_sufficient": (
        "investigate tail-risk / per-state constraints using p95/p99/max "
        "action-drift bounding rather than mean divergence"
    ),
    "radius_boundary_confirmed": (
        "fine-grain the radius threshold between 0.00125 and 0.00150 and test "
        "whether multi-step convergence within the safe radius produces incremental strength"
    ),
    "unprojected_reproduction_failure": (
        "debug the PR #204 baseline reproduction environment before proceeding"
    ),
    "projection_invariant_failure": (
        "fix projection invariant violations in the trainer or state caching"
    ),
    "inconclusive": ("expand arena sample size or evaluate additional test seeds"),
}


def replay_lane_hard_projection(
    manifest: dict[str, Any],
    workdir: Path,
    device: torch.device,
    p1_checkpoint_path: Path,
    beta: float,
    radius: float | None,
    trust_set: TrustStateSet,
    steps: list[int],
) -> tuple[
    dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    float,
    list[dict[str, Any]],
]:
    """Train one matched policy_head lane with optional hard output-space projection."""
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(paths["train_source_indexes"], allow_pickle=False)
    plan = np.load(paths["batch_indexes"], allow_pickle=False)

    batches = [
        _batch([rows[int(index)] for index in source], indexes, device)
        for indexes in plan
    ]

    model = _new_model(device)
    load_checkpoint_into_model(model, p1_checkpoint_path)
    apply_trainable_scope(model, TRAINABLE_SCOPE)

    parent = _new_model(device)
    load_checkpoint_into_model(parent, p1_checkpoint_path)
    for parameter in parent.parameters():
        parameter.requires_grad_(False)

    theta_p1 = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(manifest["optimizer"]["lr"])
    )
    saved = {0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)}

    initial_policy_grad_norm = 0.0
    step_telemetry: list[dict[str, Any]] = []
    cumulative_projected_steps = 0

    model.train()
    for step, batch in enumerate(batches, 1):
        theta_old = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}

        if beta != 0.0:
            p_parent = incumbent_policy_batch(parent, batch)
            p_beta = mixed_policy_target(batch["p"], p_parent, batch["mask"], beta)
            assert_legal_distribution(
                p_beta.detach().cpu().numpy(),
                batch["mask"].detach().cpu().numpy(),
            )
            batch = {**batch, "p": p_beta}
        policy, value = _losses(model, batch)
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()

        if step == 1:
            policy_grads = [
                p.grad
                for name, p in model.named_parameters()
                if "policy" in name and p.grad is not None
            ]
            if policy_grads:
                initial_policy_grad_norm = float(
                    torch.linalg.vector_norm(
                        torch.cat([g.flatten() for g in policy_grads])
                    )
                )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(manifest["gradient_clip"])
        )
        optimizer.step()

        # Extract raw proposed policy parameters
        theta_raw = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}

        # Hard projection onto parent-relative trust sphere
        accepted_theta, telemetry = project_policy_head_step(
            model=model,
            theta_old=theta_old,
            theta_raw=theta_raw,
            theta_p1=theta_p1,
            trust_set=trust_set,
            radius=radius,
            max_bisection_steps=30,
            tolerance=1e-6,
        )

        if telemetry["projection_activated"]:
            cumulative_projected_steps += 1

        telemetry["step"] = step
        telemetry["cumulative_projected_steps"] = cumulative_projected_steps
        telemetry["cumulative_projected_fraction"] = cumulative_projected_steps / step
        step_telemetry.append(telemetry)

        if step in steps:
            saved[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )

    if set(saved) != set(steps) | {0}:
        raise RuntimeError("failed to capture every required optimizer boundary")
    return saved, initial_policy_grad_norm, step_telemetry


def probe_target_metrics(
    probe_rows: list[dict[str, Any]],
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    p1_state: dict[str, torch.Tensor],
    p0_state: dict[str, torch.Tensor],
    beta: float,
    beta000_step46_search_ce: float | None,
) -> dict[str, Any]:
    """Compute CE, fit_fraction, and drift metrics vs P1 (parent) and P0 (incumbent)."""
    x = np.asarray([row["state"] for row in probe_rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    p_search = np.asarray([row["policy"] for row in probe_rows], dtype=np.float64)

    p1_policy, _p1_value = model_outputs(p1_state, x, mask)
    p1_policy = p1_policy.astype(np.float64)

    p0_policy, _p0_value = model_outputs(p0_state, x, mask)
    p0_policy = p0_policy.astype(np.float64)

    if beta == 0.0:
        mixed = p_search
    else:
        mixed = (1.0 - beta) * p_search + beta * p1_policy
        mixed = np.where(mask.astype(bool), mixed, 0.0)
        totals = mixed.sum(axis=1, keepdims=True)
        bad = ~np.isfinite(totals) | (totals <= 0.0)
        uniform = np.where(mask, 1.0, 0.0)
        uniform = uniform / uniform.sum(axis=1, keepdims=True)
        mixed = np.where(bad, uniform, mixed / totals)
    mixed = mixed.astype(np.float64)

    p1_search_ce = float(np.mean(_cross_entropy(p1_policy, p_search)))
    p0_search_ce = float(np.mean(_cross_entropy(p0_policy, p_search)))

    result: dict[str, Any] = {}
    for step, (state, _optimizer) in snapshots.items():
        candidate, _value = model_outputs(state, x, mask)
        candidate = candidate.astype(np.float64)

        ce_search = float(np.mean(_cross_entropy(candidate, p_search)))
        ce_p1 = float(np.mean(_cross_entropy(candidate, p1_policy)))
        ce_p0 = float(np.mean(_cross_entropy(candidate, p0_policy)))
        ce_mixed = float(np.mean(_cross_entropy(candidate, mixed)))

        improvement = p1_search_ce - ce_search
        fit_fraction: float | None
        if beta000_step46_search_ce is None:
            fit_fraction = None
        else:
            denom = p1_search_ce - beta000_step46_search_ce
            if denom <= 1e-12:
                fit_fraction = None
            else:
                fit_fraction = float(improvement / denom)

        result[str(step)] = {
            "ce_candidate_search": ce_search,
            "ce_candidate_p1": ce_p1,
            "ce_candidate_p0": ce_p0,
            "ce_candidate_mixed": ce_mixed,
            "ce_p1_search": p1_search_ce,
            "ce_p0_search": p0_search_ce,
            "search_target_ce_improvement_vs_p1": improvement,
            "fit_fraction": fit_fraction,
            "fit_denominator": (
                p1_search_ce - beta000_step46_search_ce
                if beta000_step46_search_ce is not None
                else None
            ),
            "drift_vs_p1": policy_drift_metrics(candidate, p1_policy, mask),
            "drift_vs_p0": policy_drift_metrics(candidate, p0_policy, mask),
            "drift_vs_search_target": policy_drift_metrics(candidate, p_search, mask),
        }
    return result


def paired_arena_opponent(
    artifacts: dict[str, dict[int, Path]],
    opponent: Path,
    workdir: Path,
    workers: int,
    targets: list[tuple[str, int, str]],
    role_prefix: str,
) -> dict[str, Any]:
    """Canonical paired arena evaluation of targets vs a specific opponent (P1 or P0)."""
    metrics: dict[str, Any] = {}
    control_cache: dict[str, list[dict[str, Any]]] = {}

    for lane, step, context in targets:
        if context not in control_cache:
            control_cache[context] = _arena_records(
                workdir, opponent, opponent, context, f"{role_prefix}_control", workers
            )
        control = control_cache[context]
        cand_artifact = artifacts[lane][step]
        cand_records = _arena_records(
            workdir / f"{lane}_step_{step:04d}",
            cand_artifact,
            opponent,
            context,
            f"{lane}_vs_{role_prefix}",
            workers,
        )
        effect = paired_opening_candidate_effect(cand_records, control)
        metrics.setdefault(lane, {}).setdefault(str(step), {})[context] = {
            "paired_candidate_effect": effect["paired_candidate_effect"],
            "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
            "p0_effect": effect["p0_effect"],
            "p1_effect": effect["p1_effect"],
            "win_draw_loss": _win_draw_loss(cand_records),
            "orientation": f"candidate_minus_{role_prefix}",
        }
    return metrics


def classify_hard_trust_region(summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluate prespecified classification criteria for hard trust region experiment."""
    sanity = summary.get("sanity", {})
    if (
        not sanity.get("p1_reconstructed_and_verified")
        or not sanity.get("lanes_start_identical")
        or not sanity.get("all_lanes_trunk_zero_change_vs_p1")
        or not sanity.get("all_lanes_value_stack_zero_change_vs_p1")
        or not sanity.get("all_lanes_trunk_zero_change_vs_p0")
        or not sanity.get("all_lanes_value_stack_zero_change_vs_p0")
        or not sanity.get("all_projected_lanes_respect_radius")
    ):
        label = "projection_invariant_failure"
        return {
            "label": label,
            "next_experiment": NEXT_EXPERIMENTS[label],
            "evidence": sanity,
        }

    arena = summary.get("arena", {})
    probe = summary.get("probe_target_metrics", {})
    telemetry = summary.get("projection_telemetry", {})

    vs_p1 = arena.get("vs_p1", {})
    vs_p0 = arena.get("vs_p0", {})

    # Check unprojected baseline reproduction
    unproj_384 = vs_p1.get("unprojected", {}).get("46", {}).get(STAGE1_CONTEXT)
    if unproj_384 is None or unproj_384["paired_candidate_effect"] > -0.05:
        label = "unprojected_reproduction_failure"
        return {
            "label": label,
            "next_experiment": NEXT_EXPERIMENTS[label],
            "evidence": {"unprojected_vs_p1_384": unproj_384},
        }

    # Evaluate projected lanes
    projected_lanes = ["l1_0010", "l1_00125", "l1_0015"]
    lane_safe_384: dict[str, bool] = {}
    lane_safe_1200: dict[str, bool] = {}
    lane_fit: dict[str, float] = {}
    lane_frozen_learning: dict[str, bool] = {}

    for lane in projected_lanes:
        e384 = vs_p1.get(lane, {}).get("46", {}).get(STAGE1_CONTEXT)
        lane_safe_384[lane] = e384 is not None and arena_safe(e384)

        e1200 = vs_p1.get(lane, {}).get("46", {}).get(STAGE2_CONTEXT)
        lane_safe_1200[lane] = e1200 is not None and arena_safe(e1200)

        fit = probe.get(lane, {}).get("46", {}).get("fit_fraction", 0.0)
        lane_fit[lane] = fit if fit is not None else 0.0

        # Check if lambda collapsed to ~0 on later steps (steps 30..46)
        lane_tel = telemetry.get(lane, [])
        late_lambdas = [t["lambda_accepted"] for t in lane_tel if t["step"] >= 30]
        mean_late_lambda = float(np.mean(late_lambdas)) if late_lambdas else 1.0
        lane_frozen_learning[lane] = bool(mean_late_lambda < 0.01)

    evidence = {
        "unprojected_vs_p1_384_effect": unproj_384["paired_candidate_effect"],
        "unprojected_vs_p1_384_safe": arena_safe(unproj_384),
        "lane_safe_384": lane_safe_384,
        "lane_safe_1200": lane_safe_1200,
        "lane_fit_fraction": lane_fit,
        "lane_frozen_learning": lane_frozen_learning,
    }

    # Check primary lane l1_00125
    primary_safe_384 = lane_safe_384.get("l1_00125", False)
    primary_fit = lane_fit.get("l1_00125", 0.0)

    # Check P0 benchmark for safe candidates
    p0_benchmarks_safe = True
    for lane in projected_lanes:
        if lane_safe_384.get(lane, False) and lane_fit.get(lane, 0.0) >= 0.25:
            e_p0_384 = vs_p0.get(lane, {}).get("46", {}).get(STAGE1_CONTEXT)
            e_p0_1200 = vs_p0.get(lane, {}).get("46", {}).get(STAGE2_CONTEXT)
            if e_p0_384 is not None and not arena_safe(e_p0_384):
                p0_benchmarks_safe = False
            if e_p0_1200 is not None and not arena_safe(e_p0_1200):
                p0_benchmarks_safe = False

    # Classification logic
    if (
        primary_safe_384
        and primary_fit >= 0.25
        and lane_safe_1200.get("l1_00125", False)
        and p0_benchmarks_safe
    ):
        label = "hard_projection_rescues_gen2"
    elif (
        lane_safe_384.get("l1_0010", False)
        and lane_safe_384.get("l1_00125", False)
        and not lane_safe_384.get("l1_0015", True)
    ):
        label = "radius_boundary_confirmed"
    elif any(lane_safe_384.values()) and all(
        lane_fit[lane_name] < 0.25 or lane_frozen_learning[lane_name]
        for lane_name, safe in lane_safe_384.items()
        if safe
    ):
        label = "projection_safe_but_learning_collapsed"
    elif not any(lane_safe_384.values()) and any(
        lane_fit[lane_name] >= 0.25 for lane_name in projected_lanes
    ):
        label = "mean_l1_radius_not_sufficient"
    else:
        label = "inconclusive"

    return {
        "label": label,
        "next_experiment": NEXT_EXPERIMENTS[label],
        "evidence": evidence,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render Markdown report for hard output-space trust region experiment."""
    classification = summary["classification"]
    inputs = summary["inputs"]
    sanity = summary.get("sanity") or {}
    lanes = [name for name, _ in RADIUS_LANES]
    steps = summary["checkpoint_steps"]

    lines = [
        "# AlphaZero-Lite Hard Parent-Relative Trust Region Results (PR #204 Gen-2 Update)",
        "",
        f"**Primary Classification:** `{classification['label']}`",
        "",
        f"**Recommended Next Action:** `{classification['next_experiment']}`",
        "",
        "## Core Guardrails & Invariants",
        "",
        f"- P1 reconstructed and verified: `{sanity.get('p1_reconstructed_and_verified')}`",
        f"- All lanes start from identical P1: `{sanity.get('lanes_start_identical')}`",
        f"- Trunk byte-identical to P1 (all lanes): `{sanity.get('all_lanes_trunk_zero_change_vs_p1')}`",
        f"- Trunk byte-identical to P0 (all lanes): `{sanity.get('all_lanes_trunk_zero_change_vs_p0')}`",
        f"- Value stack byte-identical to P1 (all lanes): `{sanity.get('all_lanes_value_stack_zero_change_vs_p1')}`",
        f"- Value stack byte-identical to P0 (all lanes): `{sanity.get('all_lanes_value_stack_zero_change_vs_p0')}`",
        f"- All projected lanes respect radius: `{sanity.get('all_projected_lanes_respect_radius')}`",
        f"- Trust state set size: `{inputs.get('trust_state_set', {}).get('unique_state_count')}` unique Gen-2 replay states",
        f"- Trust state set SHA256: `{inputs.get('trust_state_set', {}).get('state_set_hash')}`",
        f"- P0 weights sha256: `{inputs.get('p0_weights_sha256')}`",
        f"- P1 weights sha256: `{inputs.get('p1_weights_sha256')}`",
        f"- P1 checkpoint npz sha256: `{inputs.get('p1_checkpoint_npz_sha256')}`",
        f"- Gen-2 replay sha256: `{inputs.get('gen2_replay_sha256')}`",
        f"- Optimizer: `{json.dumps(inputs.get('optimizer'))}`",
        f"- Gradient clip: `{inputs.get('gradient_clip')}`",
        f"- Trainable scope: `{inputs.get('trainable_scope')}`",
        "",
        "## Projection Telemetry Summary",
        "",
        "| Lane | Radius | Steps Projected | Projected % | Final Raw L1 | Final Accepted L1 | Final Lambda | Adam Boundary Pressure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    telemetry = summary.get("projection_telemetry") or {}
    for lane, radius in RADIUS_LANES:
        tel = telemetry.get(lane, [])
        if not tel:
            continue
        last = tel[-1]
        proj_count = sum(1 for t in tel if t["projection_activated"])
        proj_pct = proj_count / len(tel) * 100.0
        rad_str = f"{radius:.5f}" if radius is not None else "unconstrained"
        # Check if adam pushed against boundary on later steps
        late_lambdas = [t["lambda_accepted"] for t in tel if t["step"] >= 30]
        mean_late_lambda = float(np.mean(late_lambdas)) if late_lambdas else 1.0
        pressure_str = (
            "Active (lambda < 0.20)"
            if mean_late_lambda < 0.20
            else "Moderate"
            if mean_late_lambda < 0.80
            else "Low/None"
        )
        lines.append(
            f"| {lane} | {rad_str} | {proj_count}/{len(tel)} | {proj_pct:.1f}% | "
            f"{last['raw_mean_l1']:.6f} | {last['accepted_mean_l1']:.6f} | "
            f"{last['lambda_accepted']:.4f} | {pressure_str} |"
        )

    # Detailed Step-by-Step Telemetry for Primary Lane (l1_00125)
    prim_tel = telemetry.get("l1_00125", [])
    if prim_tel:
        lines.extend(
            [
                "",
                "### Primary Lane (l1_00125) Step-by-Step Boundary Telemetry",
                "",
                "| Step | Raw Mean L1 | Accepted Mean L1 | Accepted Lambda | Projected? | Raw vs Pre-Step Delta | Acc vs Pre-Step Delta | Cumulative vs P1 Delta |",
                "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        sample_steps = [1, 2, 3, 4, 5, 6, 8, 10, 16, 20, 30, 40, 46]
        for t in prim_tel:
            if t["step"] in sample_steps:
                lines.append(
                    f"| {t['step']} | {t['raw_mean_l1']:.6f} | {t['accepted_mean_l1']:.6f} | "
                    f"{t['lambda_accepted']:.4f} | {t['projection_activated']} | "
                    f"{t['param_delta_raw_vs_old']:.6f} | {t['param_delta_acc_vs_old']:.6f} | "
                    f"{t['param_delta_acc_vs_p1']:.6f} |"
                )

    lines.extend(
        [
            "",
            "## Training & Validation Probe Metrics (Gen-2 Validation Probe)",
            "",
            "| Lane | Step | CE(search) | CE(P1) | CE(mixed) | Search-CE Improv vs P1 | Fit Fraction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    probe_metrics = summary.get("probe_target_metrics") or {}
    for lane in lanes:
        lane_metrics = probe_metrics.get(lane, {})
        for step in steps:
            entry = lane_metrics.get(str(step))
            if entry is None:
                continue
            fit = entry.get("fit_fraction")
            fit_str = f"{fit:.4f}" if fit is not None else "n/a"
            lines.append(
                f"| {lane} | {step} | {entry['ce_candidate_search']:.4f} | "
                f"{entry['ce_candidate_p1']:.4f} | {entry['ce_candidate_mixed']:.4f} | "
                f"{entry['search_target_ce_improvement_vs_p1']:+.4f} | {fit_str} |"
            )

    lines.extend(
        [
            "",
            "## Policy Drift vs Frozen P1 (Parent Reference)",
            "",
            "| Lane | Step | L1 mean | L1 max | L1 p50 | L1 p90 | L1 p95 | L1 p99 | JS mean | Top-1 Change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane in lanes:
        lane_metrics = probe_metrics.get(lane, {})
        for step in steps:
            entry = lane_metrics.get(str(step))
            if entry is None:
                continue
            d = entry.get("drift_vs_p1", {})
            lines.append(
                f"| {lane} | {step} | {d.get('legal_l1_mean', 0.0):.6f} | "
                f"{d.get('legal_l1_max', 0.0):.6f} | {d.get('legal_l1_p50', 0.0):.6f} | "
                f"{d.get('legal_l1_p90', 0.0):.6f} | {d.get('legal_l1_p95', 0.0):.6f} | "
                f"{d.get('legal_l1_p99', 0.0):.6f} | {d.get('legal_js_mean', 0.0):.6f} | "
                f"{d.get('top1_change_rate', 0.0):.4f} |"
            )

    lines.extend(
        [
            "",
            "## Cumulative Policy Drift vs P0 (Original Incumbent)",
            "",
            "| Lane | Step | L1 mean | L1 max | L1 p50 | L1 p90 | L1 p95 | L1 p99 | JS mean | Top-1 Change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane in lanes:
        lane_metrics = probe_metrics.get(lane, {})
        for step in steps:
            entry = lane_metrics.get(str(step))
            if entry is None:
                continue
            d = entry.get("drift_vs_p0", {})
            lines.append(
                f"| {lane} | {step} | {d.get('legal_l1_mean', 0.0):.6f} | "
                f"{d.get('legal_l1_max', 0.0):.6f} | {d.get('legal_l1_p50', 0.0):.6f} | "
                f"{d.get('legal_l1_p90', 0.0):.6f} | {d.get('legal_l1_p95', 0.0):.6f} | "
                f"{d.get('legal_l1_p99', 0.0):.6f} | {d.get('legal_js_mean', 0.0):.6f} | "
                f"{d.get('top1_change_rate', 0.0):.4f} |"
            )

    lines.extend(
        [
            "",
            "## Parameter Drift (Relative L2 Drift vs P1 and vs P0)",
            "",
            "| Lane | Step | Trunk (vs P1) | Policy Head (vs P1) | Value (vs P1) | Trunk (vs P0) | Policy Head (vs P0) | Value (vs P0) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    drift_p1 = summary.get("drift_vs_p1") or {}
    drift_p0 = summary.get("drift_vs_p0") or {}

    def _extract_drift(val: Any) -> float:
        return (
            float(val.get("relative_l2_norm", val))
            if isinstance(val, dict)
            else float(val)
        )

    for lane in lanes:
        for step in steps:
            e_p1 = drift_p1.get(lane, {}).get(str(step), {})
            e_p0 = drift_p0.get(lane, {}).get(str(step), {})
            t_p1 = _extract_drift(e_p1.get("trunk", 0.0))
            p_p1 = _extract_drift(e_p1.get("policy_head", 0.0))
            v_p1 = _extract_drift(e_p1.get("value_head", 0.0))
            t_p0 = _extract_drift(e_p0.get("trunk", 0.0))
            p_p0 = _extract_drift(e_p0.get("policy_head", 0.0))
            v_p0 = _extract_drift(e_p0.get("value_head", 0.0))
            lines.append(
                f"| {lane} | {step} | {t_p1:.6f} | {p_p1:.6f} | {v_p1:.6f} | {t_p0:.6f} | {p_p0:.6f} | {v_p0:.6f} |"
            )

    lines.extend(
        [
            "",
            f"## Search Diagnostics ({PUCT_CONTEXT} context, candidate vs P1)",
            "",
            "| Lane | Step | Move Change Rate | Visit JS | Q-Rank Change | Root-Value Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    puct_p1 = summary.get("puct_vs_p1") or {}
    for lane in lanes:
        metrics = puct_p1.get(lane, {}).get("metrics", {})
        for step in (16, 46):
            entry = metrics.get(str(step), {}).get(PUCT_CONTEXT)
            if entry is None:
                continue
            lines.append(
                f"| {lane} | {step} | {entry['selected_move_change_rate']:.4f} | "
                f"{entry['visit_js']:.4f} | {entry['child_q_rank_change']:+.4f} | "
                f"{entry['root_value_delta']:+.4f} |"
            )

    per_depth = summary.get("per_depth_probe_vs_p1") or {}
    if per_depth:
        lines.extend(
            [
                "",
                "## Per-Depth Policy L1/JS on Expanded Probe States "
                f"({PUCT_CONTEXT}, candidate vs P1)",
                "",
                "| Lane | Step | Depth | Expanded Nodes | L1 Mean | JS Mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for lane in ["unprojected", "l1_00125"]:
            for step in (16, 46):
                tel = (
                    per_depth.get(lane, {})
                    .get(str(step), {})
                    .get("incumbent_all", {})
                    .get("override_telemetry_by_depth")
                )
                if not tel:
                    continue
                depth_keys = sorted(
                    (d for d in tel if str(d).isdigit()), key=lambda d: int(d)
                )
                for depth in depth_keys:
                    b = tel[depth]
                    lines.append(
                        f"| {lane} | {step} | {depth} | {b['expanded_nodes']} | "
                        f"{b['mean_pairwise_legal_l1']:.6f} | "
                        f"{b['mean_pairwise_legal_js']:.6f} |"
                    )

    lines.extend(
        [
            "",
            "## Paired Arena Evaluation Matrix",
            "",
            "### 1. Candidates vs P1 (Parent Reference)",
            "",
            "| Lane | Step | Context | Paired Effect | 95% CI | P0 Effect | P1 Effect | W/D/L | Safe |",
            "| --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    arena_vs_p1 = summary.get("arena", {}).get("vs_p1") or {}
    for lane in lanes:
        lane_arena = arena_vs_p1.get(lane, {})
        for step_str in sorted(lane_arena.keys(), key=lambda s: int(s)):
            for context in (STAGE1_CONTEXT, STAGE2_CONTEXT):
                entry = lane_arena.get(step_str, {}).get(context)
                if entry is None:
                    continue
                ci = entry["opening_bootstrap_ci"]
                wdl = entry.get("win_draw_loss", {})
                is_safe = arena_safe(entry)
                lines.append(
                    f"| {lane} | {step_str} | {context} | "
                    f"{entry['paired_candidate_effect']:+.4f} | "
                    f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | "
                    f"{entry.get('p0_effect', 0.0):+.4f} | "
                    f"{entry.get('p1_effect', 0.0):+.4f} | "
                    f"{wdl.get('wins', 0)}/{wdl.get('draws', 0)}/{wdl.get('losses', 0)} | "
                    f"{is_safe} |"
                )

    arena_vs_p0 = summary.get("arena", {}).get("vs_p0") or {}
    if arena_vs_p0:
        lines.extend(
            [
                "",
                "### 2. Candidates vs P0 (Direct Cumulative Measurement)",
                "",
                "| Lane | Step | Context | Paired Effect | 95% CI | P0 Effect | P1 Effect | W/D/L | Safe |",
                "| --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- |",
            ]
        )
        for lane in arena_vs_p0:
            lane_arena = arena_vs_p0[lane]
            for step_str in sorted(lane_arena.keys(), key=lambda s: int(s)):
                for context in (STAGE1_CONTEXT, STAGE2_CONTEXT):
                    entry = lane_arena.get(step_str, {}).get(context)
                    if entry is None:
                        continue
                    ci = entry["opening_bootstrap_ci"]
                    wdl = entry.get("win_draw_loss", {})
                    is_safe = arena_safe(entry)
                    lines.append(
                        f"| {lane} | {step_str} | {context} | "
                        f"{entry['paired_candidate_effect']:+.4f} | "
                        f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | "
                        f"{entry.get('p0_effect', 0.0):+.4f} | "
                        f"{entry.get('p1_effect', 0.0):+.4f} | "
                        f"{wdl.get('wins', 0)}/{wdl.get('draws', 0)}/{wdl.get('losses', 0)} | "
                        f"{is_safe} |"
                    )

    lines.extend(
        [
            "",
            "## Classification Evidence",
            "",
            "| Signal | Value |",
            "| --- | ---: |",
        ]
    )
    for key, val in classification["evidence"].items():
        if isinstance(val, (int, float)):
            val_str = f"{val:+.4f}" if isinstance(val, float) else str(val)
        else:
            val_str = str(val)
        lines.append(f"| {key} | {val_str} |")

    lines.extend(
        [
            "",
            "## Reproduction Command",
            "",
            "```bash",
            ".venv/bin/python ml/alphazero_lite/run_gen2_hard_trust_region.py \\",
            f"  --workdir {inputs.get('workdir')} \\",
            f"  --seed {inputs.get('seed', 43)} \\",
            f"  --arena-workers {inputs.get('arena_workers', 24)}",
            "```",
            "",
            "Full JSON evidence: `docs/data/alphazero-lite-hard-parent-trust-region-summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parent-relative hard output-space trust region experiment for Gen-2 AlphaZero update."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_hard_parent_trust_region"),
        help="Experiment scratch directory.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=REPO_ROOT / "model-artifact/current",
        help="Path to P0 incumbent model-artifact/current.",
    )
    parser.add_argument(
        "--p1-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_selfplay_anchor"),
        help="Path to PR #203 P1 workdir.",
    )
    parser.add_argument(
        "--gen1-replay",
        type=Path,
        default=Path("/tmp/azlite_fresh_selfplay_anchor/fresh_self_play.jsonl"),
        help="Path to PR #203 Gen-1 replay.",
    )
    parser.add_argument(
        "--gen2-replay",
        type=Path,
        default=Path("/tmp/azlite_gen2_selfplay_anchor/gen2_self_play.jsonl"),
        help="Path to PR #204 Gen-2 replay.",
    )
    parser.add_argument("--seed", type=int, default=GEN2_DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--per-depth-probe", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-hard-parent-trust-region-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT / "docs/alphazero-lite-hard-parent-trust-region-results.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, args.seed)

    # 0. Reconstruct and freeze P1
    print(
        "[lineage] reconstructing and freezing P1 (PR #203 beta_095 step 46)...",
        flush=True,
    )
    p1_artifact_dir, p1_ckpt_npz, p1_weights_sha, p1_ckpt_sha, p1_state_hash = (
        reconstruct_and_freeze_p1(
            current_dir=args.current,
            p1_workdir=args.p1_workdir,
            gen1_replay_path=args.gen1_replay,
            workers=args.workers,
        )
    )

    p0_weights_sha = sha256_file(args.current / "weights.json")
    p0_npz = args.workdir / "p0_incumbent.npz"
    materialize_weights_json_checkpoint(
        weights_path=args.current / "weights.json",
        out_path=p0_npz,
    )

    # 1. Verify Gen-2 Replay
    if not args.gen2_replay.is_file():
        raise RuntimeError(f"Gen-2 replay not found at {args.gen2_replay}")
    gen2_replay_sha = sha256_file(args.gen2_replay)
    if gen2_replay_sha != PR204_GEN2_REPLAY_HASH:
        raise RuntimeError(
            f"Gen-2 replay SHA {gen2_replay_sha} does not match PR #204 canonical {PR204_GEN2_REPLAY_HASH}"
        )
    gen2_rows = read_jsonl(args.gen2_replay)
    print(
        f"[replay] Gen-2 replay verified: {len(gen2_rows)} rows, SHA: {gen2_replay_sha}",
        flush=True,
    )

    # 2. Manifest and Batch Plan on Gen-2 Replay
    audit_data = {
        "schema": f"{NAMESPACE}_gen2_replay_audit",
        "replay_rows": len(gen2_rows),
        "replay_sha256": gen2_replay_sha,
        "parent_weights_sha256": p1_weights_sha,
        "policy_target_mode": "default",
        "value_target_mode": "default",
        "generator": "PR #204 Gen-2 self-play replay",
    }
    (args.workdir / "gen2_replay_audit.json").write_text(
        json.dumps(audit_data, indent=2) + "\n"
    )
    build_manifest(
        rows=gen2_rows,
        workdir=args.workdir,
        current=args.current,
        replay=args.gen2_replay,
        seed=args.seed,
        epochs=1,
        batch_size=args.batch_size,
        replay_audit=args.workdir / "gen2_replay_audit.json",
    )
    manifest_path = args.workdir / "training_manifest.json"
    manifest = verify_manifest(manifest_path)

    # 3. Construct Frozen Trust-State Set
    print(
        "[trust_set] constructing deterministic trust-state set from unique Gen-2 replay states...",
        flush=True,
    )
    p1_model = _new_model(device)
    load_checkpoint_into_model(p1_model, p1_ckpt_npz)
    trust_set = TrustStateSet.from_replay_rows(gen2_rows, p1_model, device)
    print(
        f"[trust_set] trust set ready: {trust_set.unique_state_count} unique states, hash: {trust_set.state_set_hash}",
        flush=True,
    )

    p1_state = {k: v.detach().clone() for k, v in p1_model.state_dict().items()}
    p0_model = _new_model(device)
    load_checkpoint_into_model(p0_model, p0_npz)
    p0_state = {k: v.detach().clone() for k, v in p0_model.state_dict().items()}

    # 4. Train Lanes
    steps = CHECKPOINT_STEPS
    lanes: dict[str, dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]] = {}
    initial_grads: dict[str, float] = {}
    telemetry_by_lane: dict[str, list[dict[str, Any]]] = {}

    for lane, radius in RADIUS_LANES:
        rad_str = f"radius={radius}" if radius is not None else "unconstrained"
        print(
            f"[train] running lane {lane} ({rad_str}, beta=0.95, anchored to P1)...",
            flush=True,
        )
        configure_determinism(device, args.seed)
        snapshots, grad_norm, step_tel = replay_lane_hard_projection(
            manifest=manifest,
            workdir=args.workdir / lane,
            device=device,
            p1_checkpoint_path=p1_ckpt_npz,
            beta=0.95,
            radius=radius,
            trust_set=trust_set,
            steps=steps,
        )
        lanes[lane] = snapshots
        initial_grads[lane] = grad_norm
        telemetry_by_lane[lane] = step_tel

    # 5. Sanity & Invariant Verification
    print("[sanity] verifying invariants...", flush=True)
    p1_step0 = lanes["unprojected"][0][0]
    p1_state_identical = all(
        tensors_identical(lanes[lane][0][0], p1_step0) for lane, _ in RADIUS_LANES
    )

    # Trunk invariant vs P1 and vs P0
    all_trunk_zero_vs_p1 = all(
        trunk_parameters_identical(lanes[lane][step][0], p1_state)
        for lane, _ in RADIUS_LANES
        for step in steps
    )
    all_trunk_zero_vs_p0 = all(
        trunk_parameters_identical(lanes[lane][step][0], p0_state)
        for lane, _ in RADIUS_LANES
        for step in steps
    )

    # Value stack invariant vs P1 and vs P0
    all_value_zero_vs_p1 = all(
        group_parameters_identical(
            lanes[lane][step][0], p1_state, prefixes=VALUE_STACK_PREFIXES
        )
        for lane, _ in RADIUS_LANES
        for step in steps
    )
    all_value_zero_vs_p0 = all(
        group_parameters_identical(
            lanes[lane][step][0], p0_state, prefixes=VALUE_STACK_PREFIXES
        )
        for lane, _ in RADIUS_LANES
        for step in steps
    )

    # Trust region boundary invariant check for projected lanes
    projected_lanes_respect_radius = True
    for lane, radius in RADIUS_LANES:
        if radius is None:
            continue
        for step in steps:
            drift = trust_set.compute_mean_l1(
                {k: lanes[lane][step][0][k] for k in POLICY_KEYS}
            )
            if drift > radius + 1e-6:
                projected_lanes_respect_radius = False
                print(
                    f"[ERROR] Lane {lane} step {step} drift {drift:.8f} > radius {radius:.8f}",
                    flush=True,
                )

    # Verify unprojected reproduces PR #204 exactly
    unproj_s46_hash = stable_hash(
        {
            name: value.detach().cpu().numpy().tobytes().hex()
            for name, value in sorted(lanes["unprojected"][46][0].items())
        }
    )
    unprojected_reproduced = unproj_s46_hash == PR204_UNPROJECTED_S46_STATE_HASH
    print(
        f"[sanity] unprojected step46 state hash: {unproj_s46_hash} (matches PR #204: {unprojected_reproduced})",
        flush=True,
    )

    p0_post_sha = sha256_file(args.current / "weights.json")
    if p0_post_sha != CURRENT_HASH:
        raise RuntimeError(
            "FATAL: model-artifact/current was mutated during execution!"
        )

    sanity = {
        "p1_reconstructed_and_verified": True,
        "lanes_start_identical": p1_state_identical,
        "all_lanes_trunk_zero_change_vs_p1": all_trunk_zero_vs_p1,
        "all_lanes_trunk_zero_change_vs_p0": all_trunk_zero_vs_p0,
        "all_lanes_value_stack_zero_change_vs_p1": all_value_zero_vs_p1,
        "all_lanes_value_stack_zero_change_vs_p0": all_value_zero_vs_p0,
        "all_projected_lanes_respect_radius": projected_lanes_respect_radius,
        "unprojected_reproduced_pr204": unprojected_reproduced,
        "unprojected_step46_state_hash": unproj_s46_hash,
    }

    # 6. Export artifacts
    print("[artifacts] exporting immutable diagnostic artifacts...", flush=True)
    artifacts = {
        lane: export_snapshot_artifacts(lanes[lane], args.workdir / lane)
        for lane, _ in RADIUS_LANES
    }

    # 7. Training & Validation Probe Metrics
    print("[metrics] evaluating Gen-2 validation probe metrics...", flush=True)
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    validation_indexes = np.load(paths["validation_source_indexes"], allow_pickle=False)
    probe, probe_manifest = decoded_validation_manifest(gen2_rows, validation_indexes)
    probe_manifest["validation_source_indexes_sha256"] = sha256_file(
        paths["validation_source_indexes"]
    )
    probe_manifest["replay_sha256"] = gen2_replay_sha
    probe_manifest["manifest_sha256"] = stable_hash(probe_manifest)
    probe_rows = [gen2_rows[index] for index in probe_manifest["source_indexes"]]

    # Baseline denominator from PR #204 beta_000 step 46 (1.0454773386686707, denom=0.005309835780699768)
    beta000_s46_search_ce = 1.0454773386686707

    probe_metrics: dict[str, Any] = {}
    for lane, _ in RADIUS_LANES:
        probe_metrics[lane] = probe_target_metrics(
            probe_rows, lanes[lane], p1_state, p0_state, 0.95, beta000_s46_search_ce
        )

    # Drift vs P1 and vs P0
    drift_vs_p1 = {
        lane: {str(step): group_delta(lanes[lane][step][0], p1_state) for step in steps}
        for lane, _ in RADIUS_LANES
    }
    drift_vs_p0 = {
        lane: {str(step): group_delta(lanes[lane][step][0], p0_state) for step in steps}
        for lane, _ in RADIUS_LANES
    }

    # 8. Search diagnostics (PUCT probe)
    puct_vs_p1: dict[str, Any] = {}
    puct_vs_p0: dict[str, Any] = {}
    if args.puct:
        print("[puct] running PUCT search probe trajectories vs P1...", flush=True)
        probe_hash = stable_hash(probe_manifest)
        for lane, _ in RADIUS_LANES:
            puct_vs_p1[lane] = puct_trajectory(
                probe[:PROBE_SIZE],
                artifacts[lane],
                args.workdir / lane,
                probe_hash,
                contexts=(PUCT_CONTEXT,),
            )
        print(
            "[puct] running PUCT search probe trajectories for primary lane vs P0...",
            flush=True,
        )
        artifacts_p0 = {0: args.current, 46: artifacts["l1_00125"][46]}
        puct_vs_p0["l1_00125"] = puct_trajectory(
            probe[:PROBE_SIZE],
            artifacts_p0,
            args.workdir / "l1_00125_vs_p0_puct",
            probe_hash,
            contexts=(PUCT_CONTEXT,),
        )

    # 9. Per-depth probe
    per_depth_p1: dict[str, Any] = {}
    if args.per_depth_probe:
        print("[per_depth] running per-depth MCTS tree probes vs P1...", flush=True)
        for lane in ["unprojected", "l1_00125"]:
            per_depth_p1[lane] = {}
            for step in (16, 46):
                cand_path = artifacts[lane][step]
                per_depth_p1[lane][str(step)] = puct_probe(
                    probe,
                    cand_path,
                    p1_artifact_dir,
                    PUCT_CONTEXT,
                    modes=("incumbent_all",),
                )

    # 10. Paired Arena Evaluation Matrix
    arena_results: dict[str, Any] = {
        "vs_p1": {},
        "vs_p0": {},
    }
    if args.arena:
        arena_dir = args.workdir / "arena"
        arena_dir.mkdir(parents=True, exist_ok=True)

        # 1. Evaluate vs P1 @ 384:256
        print("[arena] running candidate evaluation vs P1 @ 384:256...", flush=True)
        vs_p1_384_targets = [
            ("unprojected", 46, STAGE1_CONTEXT),
            ("l1_0010", 16, STAGE1_CONTEXT),
            ("l1_0010", 46, STAGE1_CONTEXT),
            ("l1_00125", 16, STAGE1_CONTEXT),
            ("l1_00125", 46, STAGE1_CONTEXT),
            ("l1_0015", 16, STAGE1_CONTEXT),
            ("l1_0015", 46, STAGE1_CONTEXT),
        ]
        vs_p1_384 = paired_arena_opponent(
            artifacts,
            p1_artifact_dir,
            arena_dir / "vs_p1",
            args.arena_workers,
            vs_p1_384_targets,
            role_prefix="p1",
        )
        for lane, step, context in vs_p1_384_targets:
            arena_results["vs_p1"].setdefault(lane, {}).setdefault(str(step), {})[
                context
            ] = vs_p1_384[lane][str(step)][context]

        # 2. For every projected step46 lane that is arena_safe AND fit_fraction >= 0.25:
        # run candidate vs P1 @ 1200:1200, candidate vs P0 @ 384:256, candidate vs P0 @ 1200:1200
        print(
            "[arena] checking qualification for 1200:1200 and P0 benchmarks...",
            flush=True,
        )
        qualifying_lanes: list[str] = []
        for lane in ["l1_0010", "l1_00125", "l1_0015"]:
            e384 = (
                arena_results["vs_p1"].get(lane, {}).get("46", {}).get(STAGE1_CONTEXT)
            )
            fit = probe_metrics.get(lane, {}).get("46", {}).get("fit_fraction", 0.0)
            if (
                e384 is not None
                and arena_safe(e384)
                and fit is not None
                and fit >= 0.25
            ):
                qualifying_lanes.append(lane)

        print(
            f"[arena] qualifying lanes for Stage 2 & P0 evaluation: {qualifying_lanes}",
            flush=True,
        )

        if qualifying_lanes:
            # Stage 2 (1200:1200) vs P1
            vs_p1_1200_targets = [
                (lane, 46, STAGE2_CONTEXT) for lane in qualifying_lanes
            ]
            vs_p1_1200 = paired_arena_opponent(
                artifacts,
                p1_artifact_dir,
                arena_dir / "vs_p1_1200",
                args.arena_workers,
                vs_p1_1200_targets,
                role_prefix="p1",
            )
            for lane, step, context in vs_p1_1200_targets:
                arena_results["vs_p1"].setdefault(lane, {}).setdefault(str(step), {})[
                    context
                ] = vs_p1_1200[lane][str(step)][context]

            # vs P0 @ 384:256 and 1200:1200
            vs_p0_targets = []
            for lane in qualifying_lanes:
                vs_p0_targets.append((lane, 46, STAGE1_CONTEXT))
                vs_p0_targets.append((lane, 46, STAGE2_CONTEXT))

            vs_p0 = paired_arena_opponent(
                artifacts,
                args.current,
                arena_dir / "vs_p0",
                args.arena_workers,
                vs_p0_targets,
                role_prefix="p0",
            )
            for lane, step, context in vs_p0_targets:
                arena_results["vs_p0"].setdefault(lane, {}).setdefault(str(step), {})[
                    context
                ] = vs_p0[lane][str(step)][context]

    # 11. Build Summary and Classify
    summary: dict[str, Any] = {
        "schema": NAMESPACE,
        "guardrails": {
            "promotion": False,
            "runtime_mutation": False,
            "architecture_change": False,
            "value_target_change": False,
            "search_change": False,
            "diagnostic_only": True,
        },
        "inputs": {
            "workdir": str(args.workdir),
            "seed": args.seed,
            "p0_weights_sha256": p0_weights_sha,
            "p1_weights_sha256": p1_weights_sha,
            "p1_checkpoint_npz_sha256": p1_ckpt_sha,
            "p1_state_hash": p1_state_hash,
            "gen2_replay_sha256": gen2_replay_sha,
            "trust_state_set": {
                "total_replay_rows": trust_set.total_replay_rows,
                "unique_state_count": trust_set.unique_state_count,
                "state_set_hash": trust_set.state_set_hash,
            },
            "radius_lanes": [
                {"lane": lane, "radius": radius} for lane, radius in RADIUS_LANES
            ],
            "checkpoint_steps": steps,
            "trainable_scope": TRAINABLE_SCOPE,
            "optimizer": {
                "type": "Adam",
                "lr": args.lr,
                "weight_decay": 0.0,
                "optimizer_moments_unprojected": True,
            },
            "gradient_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "out_summary": str(args.out_summary),
            "out_report": str(args.out_report),
        },
        "checkpoint_steps": steps,
        "sanity": sanity,
        "projection_telemetry": telemetry_by_lane,
        "probe_target_metrics": probe_metrics,
        "drift_vs_p1": drift_vs_p1,
        "drift_vs_p0": drift_vs_p0,
        "puct_vs_p1": puct_vs_p1,
        "puct_vs_p0": puct_vs_p0,
        "per_depth_probe_vs_p1": per_depth_p1,
        "arena": arena_results,
    }

    classification = classify_hard_trust_region(summary)
    summary["classification"] = classification
    print(f"\n[classification] result: {classification['label']}", flush=True)
    print(
        f"[classification] next experiment: {classification['next_experiment']}",
        flush=True,
    )

    # 12. Write outputs
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[output] wrote JSON summary to {args.out_summary}", flush=True)

    report_md = render_markdown(summary)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(report_md, encoding="utf-8")
    print(f"[output] wrote markdown report to {args.out_report}", flush=True)


if __name__ == "__main__":
    main()
