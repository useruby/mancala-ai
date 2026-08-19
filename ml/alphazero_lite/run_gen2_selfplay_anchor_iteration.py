#!/usr/bin/env python3
# ruff: noqa: E402
"""AlphaZero Generation-2 self-play and same-state parent-relative policy anchor iteration.

Lineage:
- P0: original frozen incumbent from model-artifact/current
- P1: exact PR #203 beta_095 step-46 candidate
- P2: initialize from P1, generate fresh self-play from P1, train policy head
      only using beta=0.95 same-state anchoring to P1.

Primary question:
Does P2 remain safe versus P1 while gaining strength, and does P2 preserve or
increase the cumulative advantage versus P0?
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
    BETA_LANES,
    CHECKPOINT_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_C_PUCT,
    DEFAULT_GAMES,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LR,
    DEFAULT_SIMULATIONS,
    PROBE_SIZE,
    PUCT_CONTEXT,
    STAGE1_CONTEXT,
    STAGE2_CONTEXT,
    STAGE2_STEP,
    TRAINABLE_SCOPE,
    _batch,
    _cross_entropy,
    _losses,
    _save_snapshot,
    _win_draw_loss,
    arena_safe,
    assert_legal_distribution,
    compute_dataset_diagnostics,
    export_snapshot_artifacts,
    generate_fresh_self_play,
    group_delta,
    group_parameters_identical,
    incumbent_policy_batch,
    mixed_policy_target,
    policy_drift_metrics,
    puct_probe,
    puct_trajectory,
    replay_lane_fresh,
    state_hashes,
    tensors_identical,
    trunk_parameters_identical,
)
from ml.alphazero_lite.run_frozen_trunk_head_isolation_ablation import (  # noqa: E402
    VALUE_STACK_PREFIXES,
)
from ml.alphazero_lite.run_frozen_trunk_same_state_anchor_ablation import (  # noqa: E402
    BETA100_DRIFT_TOL,
    MEANINGFUL_FIT_FRACTION,
    NONINFERIORITY_LOWER,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    CURRENT_HASH,
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

NAMESPACE = "azlite_gen2_selfplay_anchor_v1"
GEN2_DEFAULT_SEED = 43

P0_EXPECTED_HASH = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
PR203_REPLAY_HASH = "35ac7ce9f9d596ff6a9dad27a9d2ea7c0633c7d3b84860eeeaf6e1ee78fac077"
P1_EXPECTED_STATE_HASH = (
    "a86acb54b97c860289530fcb7ca64194724d43667580a52e2948359bfa3ebdf4"
)
P1_EXPECTED_NPZ_HASH = (
    "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9"
)

NEXT_EXPERIMENTS = {
    "cumulative_safe_gain": (
        "recommend a short 3-5 generation lineage experiment with arena gating "
        "at each parent transition and a fixed P0 benchmark to detect cumulative "
        "drift (NOT implemented in this PR)"
    ),
    "safe_second_step_gain_unproven": (
        "increase evaluation power before changing beta or architecture; central "
        "uncertainty is strength signal, not training stability (NOT implemented in this PR)"
    ),
    "second_iteration_regression": (
        "compare generation-1 and generation-2 search/parent disagreement tails "
        "and per-depth drift before tuning beta (NOT implemented in this PR)"
    ),
    "anchor_learning_signal_collapsed": (
        "investigate adaptive parent-relative step sizing rather than simply "
        "lowering beta (NOT implemented in this PR)"
    ),
    "unanchored_second_generation_safe": (
        "do not assume unanchored training is generally safe; investigate "
        "replay-distribution differences between generations (NOT implemented in this PR)"
    ),
    "invariant_failure": (
        "fix the invariant failure before interpreting second-generation results"
    ),
    "inconclusive": (
        "confidence intervals do not distinguish the relevant cases; rerun with "
        "more openings or seeds"
    ),
}


def dataset_evolution_comparison(
    gen2_diag: dict[str, Any],
    gen1_diag: dict[str, Any],
) -> dict[str, Any]:
    """Compare dataset diagnostics between Generation 2 (P1) and Generation 1 (P0)."""
    return {
        "gen2_p1_selfplay": gen2_diag,
        "gen1_p0_selfplay": gen1_diag,
        "deltas_gen2_minus_gen1": {
            "game_count_delta": gen2_diag["game_count"] - gen1_diag["game_count"],
            "position_count_delta": gen2_diag["position_count"]
            - gen1_diag["position_count"],
            "game_length_mean_delta": (
                gen2_diag["game_length"]["mean"] - gen1_diag["game_length"]["mean"]
            ),
            "search_entropy_mean_delta": (
                gen2_diag["search_policy_entropy"]["mean"]
                - gen1_diag["search_policy_entropy"]["mean"]
            ),
            "parent_entropy_mean_delta": (
                gen2_diag["incumbent_policy_entropy"]["mean"]
                - gen1_diag["incumbent_policy_entropy"]["mean"]
            ),
            "l1_mean_delta": (
                gen2_diag["legal_l1_search_vs_incumbent"]["mean"]
                - gen1_diag["legal_l1_search_vs_incumbent"]["mean"]
            ),
            "js_mean_delta": (
                gen2_diag["legal_js_search_vs_incumbent"]["mean"]
                - gen1_diag["legal_js_search_vs_incumbent"]["mean"]
            ),
            "top1_disagreement_delta": (
                gen2_diag["top1_disagreement_rate"]
                - gen1_diag["top1_disagreement_rate"]
            ),
        },
    }


def replay_lane_gen2(
    manifest: dict[str, Any],
    workdir: Path,
    device: torch.device,
    p1_checkpoint_path: Path,
    beta: float,
    steps: list[int],
) -> tuple[dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]], float]:
    """Train one matched policy_head lane on Generation-2 replay data anchored to P1."""
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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(manifest["optimizer"]["lr"])
    )
    saved = {0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)}

    initial_policy_grad_norm = 0.0
    model.train()
    for step, batch in enumerate(batches, 1):
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
        if step in steps:
            saved[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )

    if set(saved) != set(steps) | {0}:
        raise RuntimeError("failed to capture every required optimizer boundary")
    return saved, initial_policy_grad_norm


def probe_target_metrics_gen2(
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


def classify_gen2(summary: dict[str, Any]) -> dict[str, Any]:
    """Prespecified decision rule for AlphaZero Generation 2 iteration."""
    sanity = summary.get("sanity", {})
    lanes_start = bool(sanity.get("lanes_start_identical"))
    trunk_zero_p1 = bool(sanity.get("all_lanes_trunk_zero_change_vs_p1"))
    value_zero_p1 = bool(sanity.get("all_lanes_value_stack_zero_change_vs_p1"))
    trunk_zero_p0 = bool(sanity.get("all_lanes_trunk_zero_change_vs_p0"))
    value_zero_p0 = bool(sanity.get("all_lanes_value_stack_zero_change_vs_p0"))
    beta100_p1_equiv = bool(sanity.get("beta_100_p1_equivalent"))
    p1_reproduced = bool(sanity.get("p1_reconstructed_and_verified"))

    if (
        not lanes_start
        or not trunk_zero_p1
        or not value_zero_p1
        or not trunk_zero_p0
        or not value_zero_p0
        or not beta100_p1_equiv
        or not p1_reproduced
    ):
        label = "invariant_failure"
        return {
            "label": label,
            "next_experiment": NEXT_EXPERIMENTS[label],
            "evidence": {
                "lanes_start_identical": lanes_start,
                "all_lanes_trunk_zero_change_vs_p1": trunk_zero_p1,
                "all_lanes_value_stack_zero_change_vs_p1": value_zero_p1,
                "all_lanes_trunk_zero_change_vs_p0": trunk_zero_p0,
                "all_lanes_value_stack_zero_change_vs_p0": value_zero_p0,
                "beta_100_p1_equivalent": beta100_p1_equiv,
                "p1_reconstructed_and_verified": p1_reproduced,
            },
        }

    arena = summary.get("arena") or {}
    probe = summary.get("probe_target_metrics") or {}
    step46 = "46"

    # Arena results vs P1
    arena_vs_p1 = arena.get("vs_p1") or {}
    beta000_p1_384 = arena_vs_p1.get("beta_000", {}).get(step46, {}).get(STAGE1_CONTEXT)
    beta095_p1_384 = arena_vs_p1.get("beta_095", {}).get(step46, {}).get(STAGE1_CONTEXT)
    beta095_p1_1200 = (
        arena_vs_p1.get("beta_095", {}).get(step46, {}).get(STAGE2_CONTEXT)
    )

    # Arena results vs P0
    arena_vs_p0 = arena.get("vs_p0") or {}
    beta095_p0_384 = arena_vs_p0.get("beta_095", {}).get(step46, {}).get(STAGE1_CONTEXT)
    beta095_p0_1200 = (
        arena_vs_p0.get("beta_095", {}).get(step46, {}).get(STAGE2_CONTEXT)
    )

    beta095_fit = probe.get("beta_095", {}).get(step46, {}).get("fit_fraction")

    beta000_p1_safe = beta000_p1_384 is not None and arena_safe(beta000_p1_384)
    beta095_p1_safe_384 = beta095_p1_384 is not None and arena_safe(beta095_p1_384)
    beta095_meaningful_fit = (
        beta095_fit is not None and beta095_fit >= MEANINGFUL_FIT_FRACTION
    )
    beta095_p1_safe_1200 = beta095_p1_1200 is not None and arena_safe(beta095_p1_1200)
    beta095_p0_safe_384 = beta095_p0_384 is not None and arena_safe(beta095_p0_384)
    beta095_p0_safe_1200 = beta095_p0_1200 is not None and arena_safe(beta095_p0_1200)

    # Positive evidence of incremental game strength vs P1 @ 1200
    beta095_p1_gain_1200 = (
        beta095_p1_1200 is not None
        and beta095_p1_1200["opening_bootstrap_ci"]["lower_95"] > 0.0
    )

    evidence = {
        "beta_100_p1_equivalent": beta100_p1_equiv,
        "beta_000_p1_step46_384_256_effect": (
            beta000_p1_384["paired_candidate_effect"] if beta000_p1_384 else None
        ),
        "beta_000_p1_step46_384_256_safe": beta000_p1_safe,
        "beta_095_p1_step46_fit_fraction": beta095_fit,
        "beta_095_p1_step46_meaningful_fit": beta095_meaningful_fit,
        "beta_095_p1_step46_384_256_effect": (
            beta095_p1_384["paired_candidate_effect"] if beta095_p1_384 else None
        ),
        "beta_095_p1_step46_384_256_safe": beta095_p1_safe_384,
        "beta_095_p1_step46_1200_1200_effect": (
            beta095_p1_1200["paired_candidate_effect"] if beta095_p1_1200 else None
        ),
        "beta_095_p1_step46_1200_1200_safe": beta095_p1_safe_1200,
        "beta_095_p1_step46_1200_1200_gain": beta095_p1_gain_1200,
        "beta_095_p0_step46_384_256_effect": (
            beta095_p0_384["paired_candidate_effect"] if beta095_p0_384 else None
        ),
        "beta_095_p0_step46_384_256_safe": beta095_p0_safe_384,
        "beta_095_p0_step46_1200_1200_effect": (
            beta095_p0_1200["paired_candidate_effect"] if beta095_p0_1200 else None
        ),
        "beta_095_p0_step46_1200_1200_safe": beta095_p0_safe_1200,
        "noninferiority_lower": NONINFERIORITY_LOWER,
        "meaningful_fit_fraction": MEANINGFUL_FIT_FRACTION,
    }

    if not beta095_p1_safe_384:
        label = "second_iteration_regression"
    elif not beta095_meaningful_fit:
        label = "anchor_learning_signal_collapsed"
    elif beta000_p1_safe:
        label = "unanchored_second_generation_safe"
    elif (
        beta095_p1_safe_384
        and beta095_p1_safe_1200
        and beta095_p0_safe_384
        and beta095_p0_safe_1200
        and beta095_meaningful_fit
        and beta095_p1_gain_1200
    ):
        label = "cumulative_safe_gain"
    elif (
        beta095_p1_safe_384
        and beta095_p1_safe_1200
        and beta095_p0_safe_384
        and beta095_p0_safe_1200
        and beta095_meaningful_fit
    ):
        label = "safe_second_step_gain_unproven"
    else:
        label = "inconclusive"

    return {
        "label": label,
        "next_experiment": NEXT_EXPERIMENTS[label],
        "evidence": evidence,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render Markdown report for Generation-2 self-play anchor iteration."""
    classification = summary["classification"]
    inputs = summary["inputs"]
    sanity = summary.get("sanity") or {}
    shift = summary.get("dataset_evolution_diagnostics") or {}
    deltas = shift.get("deltas_gen2_minus_gen1") or {}
    gen2 = shift.get("gen2_p1_selfplay") or {}
    gen1 = shift.get("gen1_p0_selfplay") or {}
    lanes = [name for name, _ in BETA_LANES]

    lines = [
        "# AlphaZero-Lite Generation-2 Self-Play Anchor Results",
        "",
        f"**Classification:** `{classification['label']}`",
        "",
        f"**Recommended Next Action:** `{classification['next_experiment']}`",
        "",
        "## Core Guardrails & Invariants",
        "",
        f"- P1 reconstructed and verified: `{sanity.get('p1_reconstructed_and_verified')}`",
        f"- all lanes start from identical P1: `{sanity.get('lanes_start_identical')}`",
        f"- trunk byte-identical to P1 (all lanes): `{sanity.get('all_lanes_trunk_zero_change_vs_p1')}`",
        f"- trunk byte-identical to P0 (all lanes): `{sanity.get('all_lanes_trunk_zero_change_vs_p0')}`",
        f"- value stack byte-identical to P1 (all lanes): `{sanity.get('all_lanes_value_stack_zero_change_vs_p1')}`",
        f"- value stack byte-identical to P0 (all lanes): `{sanity.get('all_lanes_value_stack_zero_change_vs_p0')}`",
        f"- beta_100 P1-equivalent: `{sanity.get('beta_100_p1_equivalent')}` (drift = {sanity.get('beta_100_p1_drift', 0.0):.2e}, tol = {sanity.get('beta_100_drift_tolerance', 0.001)})",
        f"- beta_100 initial policy gradient norm: `{sanity.get('beta_100_initial_policy_grad_norm', 0.0):.2e}`",
        f"- checkpoint steps: `{summary.get('checkpoint_steps')}`",
        f"- P0 weights sha256: `{inputs.get('p0_weights_sha256')}`",
        f"- P1 weights sha256: `{inputs.get('p1_weights_sha256')}`",
        f"- P1 checkpoint npz sha256: `{inputs.get('p1_checkpoint_npz_sha256')}`",
        f"- Gen-1 replay sha256: `{inputs.get('gen1_replay_sha256')}`",
        f"- Gen-2 replay sha256: `{inputs.get('gen2_selfplay', {}).get('replay_sha256')}`",
        f"- Gen-2 seed: `{inputs.get('seed')}`",
        f"- optimizer: `{json.dumps(inputs.get('optimizer'))}`",
        f"- gradient clip: `{inputs.get('gradient_clip')}`",
        f"- trainable scope: `{inputs.get('trainable_scope')}`",
        "",
        "## Dataset Evolution Diagnostics (Gen-2 P1 Replay vs Gen-1 P0 Replay)",
        "",
        "| Metric | Gen-2 (P1 Parent) | Gen-1 (P0 Parent) | Delta (Gen-2 - Gen-1) |",
        "| --- | ---: | ---: | ---: |",
        f"| Games | {gen2.get('game_count', 0)} | {gen1.get('game_count', 0)} | {deltas.get('game_count_delta', 0):+d} |",
        f"| Total Positions | {gen2.get('position_count', 0)} | {gen1.get('position_count', 0)} | {deltas.get('position_count_delta', 0):+d} |",
        f"| Game Length (mean) | {gen2.get('game_length', {}).get('mean', 0.0):.2f} | {gen1.get('game_length', {}).get('mean', 0.0):.2f} | {deltas.get('game_length_mean_delta', 0.0):+.2f} |",
        f"| Game Length (p50 / p90) | {gen2.get('game_length', {}).get('p50', 0.0):.1f} / {gen2.get('game_length', {}).get('p90', 0.0):.1f} | {gen1.get('game_length', {}).get('p50', 0.0):.1f} / {gen1.get('game_length', {}).get('p90', 0.0):.1f} | - |",
        f"| Player 0 Fraction | {gen2.get('player_distribution', {}).get('p0_fraction', 0.0):.4f} | {gen1.get('player_distribution', {}).get('p0_fraction', 0.0):.4f} | - |",
        f"| Search Policy Entropy (mean) | {gen2.get('search_policy_entropy', {}).get('mean', 0.0):.4f} | {gen1.get('search_policy_entropy', {}).get('mean', 0.0):.4f} | {deltas.get('search_entropy_mean_delta', 0.0):+.4f} |",
        f"| Parent Policy Entropy (mean) | {gen2.get('incumbent_policy_entropy', {}).get('mean', 0.0):.4f} | {gen1.get('incumbent_policy_entropy', {}).get('mean', 0.0):.4f} | {deltas.get('parent_entropy_mean_delta', 0.0):+.4f} |",
        f"| Legal L1(search, parent) (mean) | {gen2.get('legal_l1_search_vs_incumbent', {}).get('mean', 0.0):.4f} | {gen1.get('legal_l1_search_vs_incumbent', {}).get('mean', 0.0):.4f} | {deltas.get('l1_mean_delta', 0.0):+.4f} |",
        f"| Legal L1 (p50 / p90 / p95 / p99) | {gen2.get('legal_l1_search_vs_incumbent', {}).get('p50', 0.0):.4f} / {gen2.get('legal_l1_search_vs_incumbent', {}).get('p90', 0.0):.4f} / {gen2.get('legal_l1_search_vs_incumbent', {}).get('p95', 0.0):.4f} / {gen2.get('legal_l1_search_vs_incumbent', {}).get('p99', 0.0):.4f} | {gen1.get('legal_l1_search_vs_incumbent', {}).get('p50', 0.0):.4f} / {gen1.get('legal_l1_search_vs_incumbent', {}).get('p90', 0.0):.4f} / {gen1.get('legal_l1_search_vs_incumbent', {}).get('p95', 0.0):.4f} / {gen1.get('legal_l1_search_vs_incumbent', {}).get('p99', 0.0):.4f} | - |",
        f"| Legal JS Divergence (mean) | {gen2.get('legal_js_search_vs_incumbent', {}).get('mean', 0.0):.4f} | {gen1.get('legal_js_search_vs_incumbent', {}).get('mean', 0.0):.4f} | {deltas.get('js_mean_delta', 0.0):+.4f} |",
        f"| Legal JS (p50 / p90 / p95 / p99) | {gen2.get('legal_js_search_vs_incumbent', {}).get('p50', 0.0):.4f} / {gen2.get('legal_js_search_vs_incumbent', {}).get('p90', 0.0):.4f} / {gen2.get('legal_js_search_vs_incumbent', {}).get('p95', 0.0):.4f} / {gen2.get('legal_js_search_vs_incumbent', {}).get('p99', 0.0):.4f} | {gen1.get('legal_js_search_vs_incumbent', {}).get('p50', 0.0):.4f} / {gen1.get('legal_js_search_vs_incumbent', {}).get('p90', 0.0):.4f} / {gen1.get('legal_js_search_vs_incumbent', {}).get('p95', 0.0):.4f} / {gen1.get('legal_js_search_vs_incumbent', {}).get('p99', 0.0):.4f} | - |",
        f"| Top-1 Disagreement Rate | {gen2.get('top1_disagreement_rate', 0.0):.4f} | {gen1.get('top1_disagreement_rate', 0.0):.4f} | {deltas.get('top1_disagreement_delta', 0.0):+.4f} |",
        "",
        "## Training & Target Metrics (Gen-2 Validation Probe)",
        "",
        "| Lane | Step | CE(search) | CE(P1) | CE(mixed) | Search-CE Improv vs P1 | Fit Fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    probe_metrics = summary.get("probe_target_metrics") or {}
    for lane in lanes:
        lane_metrics = probe_metrics.get(lane, {})
        for step in summary["checkpoint_steps"]:
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
            "## Policy Drift vs P1 (Parent Reference)",
            "",
            "| Lane | Step | L1 mean | L1 max | L1 p50 | L1 p90 | L1 p95 | L1 p99 | JS mean | Top-1 Change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane in lanes:
        lane_metrics = probe_metrics.get(lane, {})
        for step in summary["checkpoint_steps"]:
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
        for step in summary["checkpoint_steps"]:
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
        for step in summary["checkpoint_steps"]:
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
        for step in summary["checkpoint_steps"]:
            entry = metrics.get(str(step), {}).get(PUCT_CONTEXT)
            if entry is None:
                continue
            lines.append(
                f"| {lane} | {step} | {entry['selected_move_change_rate']:.4f} | "
                f"{entry['visit_js']:.4f} | {entry['child_q_rank_change']:+.4f} | "
                f"{entry['root_value_delta']:+.4f} |"
            )

    puct_p0 = summary.get("puct_vs_p0") or {}
    if puct_p0:
        lines.extend(
            [
                "",
                f"## Cumulative Search Diagnostics ({PUCT_CONTEXT} context, candidate vs P0)",
                "",
                "| Lane | Step | Move Change Rate | Visit JS | Q-Rank Change | Root-Value Delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for lane in ["beta_095"]:
            metrics = puct_p0.get(lane, {}).get("metrics", {})
            for step in summary["checkpoint_steps"]:
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
        for lane in ["beta_095"]:
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

    per_depth_p0 = summary.get("per_depth_probe_vs_p0") or {}
    if per_depth_p0:
        lines.extend(
            [
                "",
                "## Per-Depth Policy L1/JS on Expanded Probe States "
                f"({PUCT_CONTEXT}, candidate vs P0)",
                "",
                "| Lane | Step | Depth | Expanded Nodes | L1 Mean | JS Mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for lane in ["beta_095"]:
            for step in (46,):
                tel = (
                    per_depth_p0.get(lane, {})
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
            "## Canonical Paired Arena Matrix",
            "",
            "### 1. P1 vs P0 Reproduction (PR #203 Baseline)",
            "",
            "| Match | Context | Paired Effect | 95% CI | P0 Effect | P1 Effect | W/D/L |",
            "| --- | --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    p1_vs_p0 = summary.get("arena", {}).get("p1_vs_p0") or {}
    for context in (STAGE1_CONTEXT, STAGE2_CONTEXT):
        entry = p1_vs_p0.get(context)
        if entry is None:
            continue
        ci = entry["opening_bootstrap_ci"]
        wdl = entry.get("win_draw_loss", {})
        lines.append(
            f"| P1 vs P0 | {context} | "
            f"{entry['paired_candidate_effect']:+.4f} | "
            f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | "
            f"{entry.get('p0_effect', 0.0):+.4f} | "
            f"{entry.get('p1_effect', 0.0):+.4f} | "
            f"{wdl.get('wins', 0)}/{wdl.get('draws', 0)}/{wdl.get('losses', 0)} |"
        )

    lines.extend(
        [
            "",
            "### 2. Generation-2 Candidates vs P1 (Parent)",
            "",
            "| Lane | Step | Context | Paired Effect | 95% CI | P0 Effect | P1 Effect | W/D/L |",
            "| --- | ---: | --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    arena_vs_p1 = summary.get("arena", {}).get("vs_p1") or {}
    for lane in lanes:
        for step in summary["checkpoint_steps"]:
            for context in (STAGE1_CONTEXT, STAGE2_CONTEXT):
                entry = arena_vs_p1.get(lane, {}).get(str(step), {}).get(context)
                if entry is None:
                    continue
                ci = entry["opening_bootstrap_ci"]
                wdl = entry.get("win_draw_loss", {})
                lines.append(
                    f"| {lane} | {step} | {context} | "
                    f"{entry['paired_candidate_effect']:+.4f} | "
                    f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | "
                    f"{entry.get('p0_effect', 0.0):+.4f} | "
                    f"{entry.get('p1_effect', 0.0):+.4f} | "
                    f"{wdl.get('wins', 0)}/{wdl.get('draws', 0)}/{wdl.get('losses', 0)} |"
                )

    lines.extend(
        [
            "",
            "### 3. Generation-2 Candidates vs P0 (Direct Cumulative Measurement)",
            "",
            "| Lane | Step | Context | Paired Effect | 95% CI | P0 Effect | P1 Effect | W/D/L |",
            "| --- | ---: | --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    arena_vs_p0 = summary.get("arena", {}).get("vs_p0") or {}
    for lane in ["beta_095"]:
        for step in (46,):
            for context in (STAGE1_CONTEXT, STAGE2_CONTEXT):
                entry = arena_vs_p0.get(lane, {}).get(str(step), {}).get(context)
                if entry is None:
                    continue
                ci = entry["opening_bootstrap_ci"]
                wdl = entry.get("win_draw_loss", {})
                lines.append(
                    f"| {lane} | {step} | {context} | "
                    f"{entry['paired_candidate_effect']:+.4f} | "
                    f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | "
                    f"{entry.get('p0_effect', 0.0):+.4f} | "
                    f"{entry.get('p1_effect', 0.0):+.4f} | "
                    f"{wdl.get('wins', 0)}/{wdl.get('draws', 0)}/{wdl.get('losses', 0)} |"
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
            "## Exact Reproduction Commands",
            "",
            "```bash",
            ".venv/bin/python ml/alphazero_lite/run_gen2_selfplay_anchor_iteration.py \\",
            f"  --workdir {inputs.get('workdir')} \\",
            f"  --games {inputs.get('gen2_selfplay', {}).get('games_requested', 700)} \\",
            f"  --seed {inputs.get('seed', 43)} \\",
            "  --arena-workers 24",
            "```",
            "",
            "Full JSON evidence: `docs/data/alphazero-lite-gen2-selfplay-anchor-summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaZero Generation-2 self-play and parent-anchored policy update."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_gen2_selfplay_anchor"),
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
        help="Path to PR #203 P1 workdir (reconstructed if not present).",
    )
    parser.add_argument(
        "--gen1-replay",
        type=Path,
        default=Path("/tmp/azlite_fresh_selfplay_anchor/fresh_self_play.jsonl"),
        help="Path to PR #203 Gen-1 replay.",
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--seed", type=int, default=GEN2_DEFAULT_SEED)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--c-puct", type=float, default=DEFAULT_C_PUCT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument("--reuse-gen2-selfplay", action="store_true")
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--per-depth-probe", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-gen2-selfplay-anchor-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT / "docs/alphazero-lite-gen2-selfplay-anchor-results.md",
    )
    return parser.parse_args()


def reconstruct_and_freeze_p1(
    current_dir: Path,
    p1_workdir: Path,
    gen1_replay_path: Path,
    workers: int,
) -> tuple[Path, Path, str, str, str]:
    """Deterministically reconstruct and freeze P1, returning (p1_artifact_dir, p1_ckpt_npz, p1_weights_sha, p1_ckpt_sha, p1_state_hash)."""
    p1_workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # Verify P0
    p0_weights_sha = sha256_file(current_dir / "weights.json")
    if p0_weights_sha != CURRENT_HASH:
        raise RuntimeError(
            f"current artifact weights {p0_weights_sha} do not match expected {CURRENT_HASH}"
        )

    # Materialize P0 npz
    p0_npz = materialize_weights_json_checkpoint(
        weights_path=current_dir / "weights.json",
        out_path=p1_workdir / "p0_initialization.npz",
    )

    # Check / generate Gen-1 replay
    if not gen1_replay_path.is_file():
        print(
            "[p1_reconstruct] generating Gen-1 replay from P0 with seed 42...",
            flush=True,
        )
        meta = generate_fresh_self_play(
            out_path=gen1_replay_path,
            checkpoint_npz=p0_npz,
            games=DEFAULT_GAMES,
            seed=42,
            simulations=DEFAULT_SIMULATIONS,
            c_puct=DEFAULT_C_PUCT,
            workers=workers,
        )
        gen1_sha = meta["replay_sha256"]
    else:
        gen1_sha = sha256_file(gen1_replay_path)

    if gen1_sha != PR203_REPLAY_HASH:
        raise RuntimeError(
            f"Gen-1 replay SHA {gen1_sha} does not match PR #203 canonical {PR203_REPLAY_HASH}"
        )

    # Check if P1 snapshot already exists and is verified
    p1_step46_artifact = p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    p1_step46_npz = p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"

    if (
        not (p1_step46_artifact / "weights.json").is_file()
        or not p1_step46_npz.is_file()
    ):
        print(
            "[p1_reconstruct] training P1 (beta_095) from P0 on Gen-1 replay...",
            flush=True,
        )
        gen1_rows = read_jsonl(gen1_replay_path)
        audit_data = {
            "schema": f"{NAMESPACE}_gen1_audit",
            "replay_rows": len(gen1_rows),
            "replay_sha256": gen1_sha,
            "current_weights_sha256": CURRENT_HASH,
            "policy_target_mode": "default",
            "value_target_mode": "default",
            "generator": "self_play.py fresh 384-simulation PUCT",
        }
        (p1_workdir / "gen1_audit.json").write_text(
            json.dumps(audit_data, indent=2) + "\n"
        )
        build_manifest(
            rows=gen1_rows,
            workdir=p1_workdir,
            current=current_dir,
            replay=gen1_replay_path,
            seed=42,
            epochs=1,
            batch_size=DEFAULT_BATCH_SIZE,
            replay_audit=p1_workdir / "gen1_audit.json",
        )
        manifest = verify_manifest(p1_workdir / "training_manifest.json")
        snapshots, _initial_grad = replay_lane_fresh(
            manifest=manifest,
            workdir=p1_workdir / "beta_095",
            device=device,
            beta=0.95,
            steps=CHECKPOINT_STEPS,
        )
        export_snapshot_artifacts(snapshots, p1_workdir / "beta_095")

    # Verify P1 state hash
    p1_model = _new_model(device)
    load_checkpoint_into_model(p1_model, p1_step46_npz)
    p1_state_hash = stable_hash(
        {
            name: value.detach().cpu().numpy().tobytes().hex()
            for name, value in p1_model.state_dict().items()
        }
    )
    if p1_state_hash != P1_EXPECTED_STATE_HASH:
        raise RuntimeError(
            f"reconstructed P1 state hash {p1_state_hash} != expected {P1_EXPECTED_STATE_HASH}"
        )

    p1_ckpt_sha = sha256_file(p1_step46_npz)
    p1_weights_sha = sha256_file(p1_step46_artifact / "weights.json")

    print(
        f"[p1_reconstruct] P1 verified! weights_sha: {p1_weights_sha}, npz_sha: {p1_ckpt_sha}, state_hash: {p1_state_hash}",
        flush=True,
    )
    return p1_step46_artifact, p1_step46_npz, p1_weights_sha, p1_ckpt_sha, p1_state_hash


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

    # 1. Generate Generation-2 fresh self-play from P1
    gen2_replay_path = args.workdir / "gen2_self_play.jsonl"
    if args.reuse_gen2_selfplay and gen2_replay_path.is_file():
        print(
            f"[self_play] reusing existing Gen-2 self-play: {gen2_replay_path}",
            flush=True,
        )
        gen2_rows = read_jsonl(gen2_replay_path)
        game_indices = set(int(r["game_index"]) for r in gen2_rows if "game_index" in r)
        gen2_selfplay_meta = {
            "games_requested": args.games,
            "games_generated": len(game_indices),
            "positions_generated": len(gen2_rows),
            "seed": args.seed,
            "simulations": args.simulations,
            "c_puct": args.c_puct,
            "player_mode": "puct",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "default",
            "value_target_mode": "default",
            "policy_target_noise_mode": "noisy",
            "tree_reuse_enabled": True,
            "replay_path": str(gen2_replay_path),
            "replay_sha256": sha256_file(gen2_replay_path),
            "checkpoint_npz_sha256": sha256_file(p1_ckpt_npz),
        }
    else:
        print(
            f"[self_play] generating Gen-2 self-play from P1 with seed {args.seed}...",
            flush=True,
        )
        gen2_selfplay_meta = generate_fresh_self_play(
            out_path=gen2_replay_path,
            checkpoint_npz=p1_ckpt_npz,
            games=args.games,
            seed=args.seed,
            simulations=args.simulations,
            c_puct=args.c_puct,
            workers=args.workers,
        )
        gen2_rows = read_jsonl(gen2_replay_path)

    # 2. Manifest and Batch Plan on Gen-2 Replay
    audit_data = {
        "schema": f"{NAMESPACE}_gen2_replay_audit",
        "replay_rows": len(gen2_rows),
        "replay_sha256": sha256_file(gen2_replay_path),
        "parent_weights_sha256": p1_weights_sha,
        "policy_target_mode": "default",
        "value_target_mode": "default",
        "generator": f"self_play.py fresh {args.simulations}-simulation PUCT from P1",
    }
    (args.workdir / "gen2_replay_audit.json").write_text(
        json.dumps(audit_data, indent=2) + "\n"
    )
    build_manifest(
        rows=gen2_rows,
        workdir=args.workdir,
        current=args.current,
        replay=gen2_replay_path,
        seed=args.seed,
        epochs=1,
        batch_size=args.batch_size,
        replay_audit=args.workdir / "gen2_replay_audit.json",
    )
    manifest_path = args.workdir / "training_manifest.json"
    manifest = verify_manifest(manifest_path)

    # 3. Dataset evolution diagnostics (Gen-2 vs Gen-1)
    print("[diagnostics] computing dataset evolution diagnostics...", flush=True)
    p1_model = _new_model(device)
    load_checkpoint_into_model(p1_model, p1_ckpt_npz)
    p1_state = {k: v.detach().clone() for k, v in p1_model.state_dict().items()}

    p0_model = _new_model(device)
    load_checkpoint_into_model(p0_model, p0_npz)
    p0_state = {k: v.detach().clone() for k, v in p0_model.state_dict().items()}

    gen2_diag = compute_dataset_diagnostics(gen2_rows, p1_state)
    gen1_rows = read_jsonl(args.gen1_replay)
    gen1_diag = compute_dataset_diagnostics(gen1_rows, p0_state)
    evolution_diag = dataset_evolution_comparison(gen2_diag, gen1_diag)

    # 4. Matched Gen-2 training lanes
    steps = CHECKPOINT_STEPS
    lanes: dict[str, dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]] = {}
    initial_grads: dict[str, float] = {}

    for lane, beta in BETA_LANES:
        print(
            f"[train] running Gen-2 lane {lane} (beta={beta}, anchored to P1)...",
            flush=True,
        )
        configure_determinism(device, args.seed)
        snapshots, grad_norm = replay_lane_gen2(
            manifest=manifest,
            workdir=args.workdir / lane,
            device=device,
            p1_checkpoint_path=p1_ckpt_npz,
            beta=beta,
            steps=steps,
        )
        lanes[lane] = snapshots
        initial_grads[lane] = grad_norm

    # 5. Sanity & Invariant Verification
    print("[sanity] verifying Gen-2 invariants...", flush=True)
    p1_step0 = lanes["beta_000"][0][0]
    p1_state_identical = all(
        tensors_identical(lanes[lane][0][0], p1_step0) for lane, _ in BETA_LANES
    )

    # Trunk invariant vs P1 and vs P0 (must be identical)
    all_trunk_zero_vs_p1 = all(
        trunk_parameters_identical(lanes[lane][step][0], p1_state)
        for lane, _ in BETA_LANES
        for step in steps
    )
    all_trunk_zero_vs_p0 = all(
        trunk_parameters_identical(lanes[lane][step][0], p0_state)
        for lane, _ in BETA_LANES
        for step in steps
    )

    # Value invariant vs P1 and vs P0 (must be identical)
    all_value_zero_vs_p1 = all(
        group_parameters_identical(
            lanes[lane][step][0], p1_state, prefixes=VALUE_STACK_PREFIXES
        )
        for lane, _ in BETA_LANES
        for step in steps
    )
    all_value_zero_vs_p0 = all(
        group_parameters_identical(
            lanes[lane][step][0], p0_state, prefixes=VALUE_STACK_PREFIXES
        )
        for lane, _ in BETA_LANES
        for step in steps
    )

    beta100_drift_vs_p1 = group_delta(lanes["beta_100"][46][0], p1_state)["policy_head"]
    beta100_p1_equivalent = bool(beta100_drift_vs_p1 <= BETA100_DRIFT_TOL)

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
        "beta_100_p1_equivalent": beta100_p1_equivalent,
        "beta_100_p1_drift": beta100_drift_vs_p1,
        "beta_100_drift_tolerance": BETA100_DRIFT_TOL,
        "beta_100_initial_policy_grad_norm": initial_grads.get("beta_100", 0.0),
    }

    # 6. Export Gen-2 artifacts
    print("[artifacts] exporting immutable diagnostic artifacts...", flush=True)
    artifacts = {
        lane: export_snapshot_artifacts(lanes[lane], args.workdir / lane)
        for lane, _ in BETA_LANES
    }

    # 7. Training & Validation Probe Metrics
    print("[metrics] evaluating Gen-2 validation probe metrics...", flush=True)
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    validation_indexes = np.load(paths["validation_source_indexes"], allow_pickle=False)
    probe, probe_manifest = decoded_validation_manifest(gen2_rows, validation_indexes)
    probe_manifest["validation_source_indexes_sha256"] = sha256_file(
        paths["validation_source_indexes"]
    )
    probe_manifest["replay_sha256"] = sha256_file(gen2_replay_path)
    probe_manifest["manifest_sha256"] = stable_hash(probe_manifest)
    probe_rows = [gen2_rows[index] for index in probe_manifest["source_indexes"]]

    # Pass 1: beta_000 step 46 to find baseline denominator
    beta000_temp = probe_target_metrics_gen2(
        probe_rows, lanes["beta_000"], p1_state, p0_state, 0.0, None
    )
    beta000_s46_search_ce = beta000_temp["46"]["ce_candidate_search"]

    # Pass 2: compute fit fraction for all lanes
    probe_metrics: dict[str, Any] = {}
    for lane, beta in BETA_LANES:
        probe_metrics[lane] = probe_target_metrics_gen2(
            probe_rows, lanes[lane], p1_state, p0_state, beta, beta000_s46_search_ce
        )

    # Drift vs P1 and vs P0
    drift_vs_p1 = {
        lane: {str(step): group_delta(lanes[lane][step][0], p1_state) for step in steps}
        for lane, _ in BETA_LANES
    }

    drift_vs_p0 = {
        lane: {str(step): group_delta(lanes[lane][step][0], p0_state) for step in steps}
        for lane, _ in BETA_LANES
    }

    # 8. Search diagnostics (PUCT probe)
    puct_vs_p1: dict[str, Any] = {}
    puct_vs_p0: dict[str, Any] = {}
    if args.puct:
        print("[puct] running PUCT search probe trajectories vs P1...", flush=True)
        probe_hash = stable_hash(probe_manifest)
        for lane, _beta in BETA_LANES:
            puct_vs_p1[lane] = puct_trajectory(
                probe[:PROBE_SIZE],
                artifacts[lane],
                args.workdir / lane,
                probe_hash,
                contexts=(PUCT_CONTEXT,),
            )
        print(
            "[puct] running PUCT search probe trajectories for beta_095 vs P0...",
            flush=True,
        )
        artifacts_p0 = {0: args.current, 46: artifacts["beta_095"][46]}
        puct_vs_p0["beta_095"] = puct_trajectory(
            probe[:PROBE_SIZE],
            artifacts_p0,
            args.workdir / "beta_095_vs_p0_puct",
            probe_hash,
            contexts=(PUCT_CONTEXT,),
        )

    # 9. Per-depth probe
    per_depth_p1: dict[str, Any] = {}
    per_depth_p0: dict[str, Any] = {}
    if args.per_depth_probe:
        print("[per_depth] running per-depth MCTS tree probes vs P1...", flush=True)
        for lane in ["beta_095"]:
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
        print(
            "[per_depth] running per-depth MCTS tree probes for beta_095 step 46 vs P0...",
            flush=True,
        )
        per_depth_p0["beta_095"] = {
            "46": puct_probe(
                probe,
                artifacts["beta_095"][46],
                args.current,
                PUCT_CONTEXT,
                modes=("incumbent_all",),
            )
        }

    # 10. Paired Arena Evaluation Matrix
    arena_results: dict[str, Any] = {
        "p1_vs_p0": {},
        "vs_p1": {},
        "vs_p0": {},
    }
    if args.arena:
        arena_dir = args.workdir / "arena"
        arena_dir.mkdir(parents=True, exist_ok=True)

        # 1. Reproduce P1 vs P0
        print("[arena] reproducing P1 vs P0 (384:256 and 1200:1200)...", flush=True)
        p1_repro_targets = [("p1", 0, STAGE1_CONTEXT), ("p1", 0, STAGE2_CONTEXT)]
        p1_repro_results = paired_arena_opponent(
            {"p1": {0: p1_artifact_dir}},
            args.current,
            arena_dir / "p1_vs_p0",
            args.arena_workers,
            p1_repro_targets,
            role_prefix="p0",
        )
        arena_results["p1_vs_p0"] = {
            STAGE1_CONTEXT: p1_repro_results["p1"]["0"][STAGE1_CONTEXT],
            STAGE2_CONTEXT: p1_repro_results["p1"]["0"][STAGE2_CONTEXT],
        }

        # 2. Evaluate Gen-2 Candidates vs P1 & P0: Stage 1 (384:256)
        print("[arena] running Gen-2 candidates vs P1 @ 384:256...", flush=True)
        vs_p1_stage1_targets = [
            ("beta_000", 46, STAGE1_CONTEXT),
            ("beta_095", 16, STAGE1_CONTEXT),
            ("beta_095", 46, STAGE1_CONTEXT),
            ("beta_100", 46, STAGE1_CONTEXT),
        ]
        vs_p1_stage1 = paired_arena_opponent(
            artifacts,
            p1_artifact_dir,
            arena_dir / "vs_p1",
            args.arena_workers,
            vs_p1_stage1_targets,
            role_prefix="p1",
        )
        for lane, step, context in vs_p1_stage1_targets:
            arena_results["vs_p1"].setdefault(lane, {}).setdefault(str(step), {})[
                context
            ] = vs_p1_stage1[lane][str(step)][context]

        print("[arena] running DIRECT measurement: P2 vs P0 @ 384:256...", flush=True)
        vs_p0_stage1_targets = [
            ("beta_095", 46, STAGE1_CONTEXT),
        ]
        vs_p0_stage1 = paired_arena_opponent(
            artifacts,
            args.current,
            arena_dir / "vs_p0",
            args.arena_workers,
            vs_p0_stage1_targets,
            role_prefix="p0",
        )
        for lane, step, context in vs_p0_stage1_targets:
            arena_results["vs_p0"].setdefault(lane, {}).setdefault(str(step), {})[
                context
            ] = vs_p0_stage1[lane][str(step)][context]

        # Stage 2 Gate: If beta_095 step 46 passes 384:256 safety gate vs P1
        entry_095_p1_384 = (
            arena_results["vs_p1"]
            .get("beta_095", {})
            .get(str(STAGE2_STEP), {})
            .get(STAGE1_CONTEXT)
        )
        fit_095 = (
            probe_metrics.get("beta_095", {})
            .get(str(STAGE2_STEP), {})
            .get("fit_fraction")
        )

        p2_passes_stage1 = (
            entry_095_p1_384 is not None
            and fit_095 is not None
            and arena_safe(entry_095_p1_384)
            and fit_095 >= MEANINGFUL_FIT_FRACTION
        )

        if args.stage2 and p2_passes_stage1:
            print(
                "[arena] beta_095 step 46 PASSED Stage 1! Running Stage 2 (1200:1200) vs P1...",
                flush=True,
            )
            vs_p1_stage2_targets = [("beta_095", STAGE2_STEP, STAGE2_CONTEXT)]
            vs_p1_stage2 = paired_arena_opponent(
                artifacts,
                p1_artifact_dir,
                arena_dir / "vs_p1",
                args.arena_workers,
                vs_p1_stage2_targets,
                role_prefix="p1",
            )
            arena_results["vs_p1"]["beta_095"][str(STAGE2_STEP)][STAGE2_CONTEXT] = (
                vs_p1_stage2["beta_095"][str(STAGE2_STEP)][STAGE2_CONTEXT]
            )

            # DIRECTLY measure P2 vs P0 at 1200:1200
            print(
                "[arena] running DIRECT measurement: P2 vs P0 @ 1200:1200...",
                flush=True,
            )
            vs_p0_stage2_targets = [
                ("beta_095", STAGE2_STEP, STAGE2_CONTEXT),
            ]
            vs_p0_stage2 = paired_arena_opponent(
                artifacts,
                args.current,
                arena_dir / "vs_p0",
                args.arena_workers,
                vs_p0_stage2_targets,
                role_prefix="p0",
            )
            for lane, step, context in vs_p0_stage2_targets:
                arena_results["vs_p0"].setdefault(lane, {}).setdefault(str(step), {})[
                    context
                ] = vs_p0_stage2[lane][str(step)][context]
        else:
            print(
                "[arena] beta_095 step 46 did not pass Stage 1 safety/fit gate for 1200:1200 arena.",
                flush=True,
            )

    # 11. Build Full Summary and Classify
    summary: dict[str, Any] = {
        "schema": NAMESPACE,
        "guardrails": {
            "promotion": False,
            "runtime_mutation": False,
            "architecture_change": False,
            "value_target_change": False,
            "search_change": False,
            "fresh_self_play": True,
            "diagnostic_only": True,
        },
        "inputs": {
            "workdir": str(args.workdir),
            "p0_weights_sha256": p0_weights_sha,
            "p1_weights_sha256": p1_weights_sha,
            "p1_checkpoint_npz_sha256": p1_ckpt_sha,
            "p1_state_hash": p1_state_hash,
            "gen1_replay_sha256": sha256_file(args.gen1_replay),
            "gen2_selfplay": gen2_selfplay_meta,
            "seed": args.seed,
            "beta_lanes": [
                {"lane": lane_name, "beta": b} for lane_name, b in BETA_LANES
            ],
            "checkpoint_steps": steps,
            "trainable_scope": TRAINABLE_SCOPE,
            "lane_trainable_scope": {
                lane_name: TRAINABLE_SCOPE for lane_name, _ in BETA_LANES
            },
            "optimizer": {"type": "Adam", "lr": args.lr, "weight_decay": 0.0},
            "gradient_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "out_summary": str(args.out_summary),
            "out_report": str(args.out_report),
        },
        "sanity": sanity,
        "dataset_evolution_diagnostics": evolution_diag,
        "checkpoint_steps": steps,
        "probe_manifest": probe_manifest,
        "probe_target_metrics": probe_metrics,
        "drift_vs_p1": drift_vs_p1,
        "drift_vs_p0": drift_vs_p0,
        "state_hashes": {
            lane: state_hashes(snapshots) for lane, snapshots in lanes.items()
        },
        "puct_vs_p1": puct_vs_p1,
        "puct_vs_p0": puct_vs_p0,
        "per_depth_probe_vs_p1": per_depth_p1,
        "per_depth_probe_vs_p0": per_depth_p0,
        "arena": arena_results,
    }

    classification = classify_gen2(summary)
    summary["classification"] = classification

    # Write summary JSON and markdown report
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[out] committed JSON summary written: {args.out_summary}", flush=True)

    markdown = render_markdown(summary)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(markdown)
    print(f"[out] markdown report written: {args.out_report}", flush=True)
    print(f"[classification] {classification['label']}", flush=True)


if __name__ == "__main__":
    main()
