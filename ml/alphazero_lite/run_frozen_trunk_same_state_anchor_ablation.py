#!/usr/bin/env python3
# ruff: noqa: E402
"""Same-state incumbent-anchored policy-target ablation (PR #200/#201 follow-up).

PR #200 isolated the frozen-trunk low-budget regression to the policy head
(``policy_head_accumulation``); PR #201 localized it to *distributed prior
compounding* throughout MCTS (root-only substitution recovered 0%, full-tree
substitution recovered 100%). PR #201's recommended next action was a global
policy trust region on legal-policy divergence from the incumbent, applied to
every replayed state at all depths. This experiment tests that recommendation in
its minimal same-state form: the search policy target ``p_search(x)`` on every
replay training state ``x`` is mixed with the *frozen incumbent's* own
legal-normalized policy on that exact same state ``p_inc(x)``:

    p_beta(x) = (1 - beta) * p_search(x) + beta * p_inc(x)

``beta`` is the incumbent-anchor fraction. Only the policy head is trained; the
trunk and value stack remain frozen (byte-identical to the incumbent) via the
existing ``policy_head`` trainable scope. The value target, optimizer, learning
rate, gradient clipping, seed, batch plan, replay rows, and initialization
checkpoint are byte-identical to PR #200's ``policy_head`` lane; only the policy
target is a same-state mixture.

Lanes:

- ``beta_000`` (beta=0.00): pure search target; must reproduce PR #200
  ``policy_head`` byte-for-byte.
- ``beta_050`` (beta=0.50), ``beta_080`` (0.80), ``beta_095`` (0.95): mixed
  targets.
- ``beta_100`` (beta=1.00): pure incumbent policy; the policy-head gradient is
  ~0 at initialization, so the lane must remain search-equivalent to the
  incumbent within numerical tolerance.

The target is constructed inside this focused runner over the exact PR #199/#200
fixed-batch replay. No general ``train.py`` policy-target mode is added, no
separate behavior-anchor dataset is used (the incumbent target is computed on
the exact same training state as the search target), no self-play is
regenerated, no architecture/value-target/search change is made, and nothing is
promoted.
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
    score_from_game,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_frozen_trunk_distillation_ablation import (  # noqa: E402
    checkpoint_steps,
    group_delta,
    lane_trainable_scope,
    state_hashes,
    tensors_identical,
    trunk_parameters_identical,
    with_total_loss,
)
from ml.alphazero_lite.run_frozen_trunk_head_isolation_ablation import (  # noqa: E402
    VALUE_STACK_PREFIXES,
    group_parameters_identical,
    probe_output_drift,
)
from ml.alphazero_lite.run_frozen_trunk_policy_prior_localization import (  # noqa: E402
    puct_probe,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    CURRENT_HASH,
    _batch,
    _losses,
    _new_model,
    _save_snapshot,
    export_snapshot_artifacts,
    output_drift,
    puct_trajectory,
)
from ml.alphazero_lite.run_policy_detached_trunk_ablation import _arena_records  # noqa: E402
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    decoded_validation_manifest,
    js,
    model_outputs,
    stable_hash,
)
from ml.alphazero_lite.train import (  # noqa: E402
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

NAMESPACE = "azlite_frozen_trunk_same_state_anchor_v1"
# (lane, beta); beta is the incumbent-anchor fraction in p_beta.
BETA_LANES: tuple[tuple[str, float], ...] = (
    ("beta_000", 0.00),
    ("beta_050", 0.50),
    ("beta_080", 0.80),
    ("beta_095", 0.95),
    ("beta_100", 1.00),
)
TRAINABLE_SCOPE = "policy_head"
PRIMARY_CONTEXT = "384:256"
PUCT_CONTEXT = "384:256"
STAGE1_CONTEXT = "384:256"
STAGE2_CONTEXT = "1200:1200"
STAGE1_STEPS = (16, 46)
STAGE2_STEP = 46
PROBE_SIZE = 256
# Prespecified decision thresholds (not loosened after seeing results).
NONINFERIORITY_LOWER = -0.03
MEANINGFUL_FIT_FRACTION = 0.25
# beta_100 search-equivalence tolerance: the policy-head relative L2 drift from
# the incumbent must stay below this for the lane to count as incumbent-equivalent.
# The pure-incumbent target yields an approximately-zero policy gradient (the
# candidate softmax equals the target p_inc up to float32 rounding), so the
# observed numerical-drift floor over 46 steps is ~4e-5; 1e-3 is comfortably above
# that floor and far below any genuinely-anchored (beta < 1) lane's drift.
BETA100_DRIFT_TOL = 1e-3

# PR #200 (commit 11f5623) recorded ``policy_head`` state hashes; beta_000 must
# reproduce them byte-for-byte before any anchor result is read.
PR200_POLICY_HEAD_STATE_HASHES = {
    "0": "d265537d6b637b8433b093ebb1f9d55fef25b38e259ed7e59f7a597b35bb6f02",
    "1": "877b6054dd4f52713a0439f10262434987439f05674838f03e16b593904a3aec",
    "4": "5c68641a66ea8633c792e89f8af06e8b7f221b6f5cd6033691493b2164247ea9",
    "16": "ec77d57c3b192edd9416f854dbfb1daf8e9675c1a8b1b8476814f3cf35408a33",
    "46": "ee12bcd171ee95f54fd4d400c955d9469a7eca565995eeebd566a4a56dcba4e0",
}


def mixed_policy_target(
    p_search: torch.Tensor,
    p_incumbent: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Construct p_beta = (1-beta) p_search + beta p_incumbent, legal-renormalized.

    ``mask`` is the float legal mask used by the policy loss (1.0 legal, 0.0
    illegal). Illegal mass is zeroed and the result is renormalized over legal
    moves. A defensive uniform-legal fallback is used wherever the legal total
    is non-finite or non-positive. ``beta == 0`` returns ``p_search`` untouched
    (byte-identical to the PR #200 search target) so that the beta_000 lane
    reproduces PR #200 exactly.
    """
    if beta == 0.0:
        return p_search
    legal = (mask > 0).to(p_search.dtype)
    mixed = (1.0 - beta) * p_search + beta * p_incumbent
    mixed = torch.where(mask > 0, mixed, torch.zeros_like(mixed))
    totals = mixed.sum(dim=1, keepdim=True)
    bad = ~torch.isfinite(totals) | (totals <= 0.0)
    uniform = legal / legal.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.where(bad, uniform, mixed / totals.clamp_min(1e-12)).to(p_search.dtype)


def assert_legal_distribution(p: np.ndarray, mask: np.ndarray) -> None:
    """Defensive assertions: finite, no illegal mass, legal sum ~= 1."""
    if not np.all(np.isfinite(p)):
        raise AssertionError("non-finite probability in mixed target")
    illegal = float(np.sum(p[~mask.astype(bool)]))
    if illegal > 1e-6:
        raise AssertionError(f"illegal probability mass in mixed target: {illegal}")
    sums = p.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise AssertionError(f"mixed target not normalized: sums={sums}")


def incumbent_policy_batch(
    incumbent: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Legal-normalized incumbent policy over exactly the batch states.

    The masking matches the policy-loss masking exactly: logits are
    ``masked_fill(mask <= 0, -1e9)`` before ``softmax`` (the policy loss uses
    the same masked logits under ``log_softmax``).
    """
    incumbent.eval()
    with torch.no_grad():
        logits, _ = incumbent(batch["x"])
    return torch.softmax(logits.masked_fill(batch["mask"] <= 0, -1e9), dim=1)


def replay_lane_anchored(
    manifest: dict[str, Any],
    workdir: Path,
    device: torch.device,
    beta: float,
    steps: list[int],
) -> dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]:
    """Replay the saved PR #191 batch plan with the same-state mixed target.

    Identical to PR #200's ``replay_lane(..., "policy_head", steps)`` except
    that for ``beta > 0`` each batch's policy target is replaced by
    ``p_beta = (1-beta) p_search + beta p_inc`` computed from the frozen
    incumbent on the exact same batch states. For ``beta == 0`` the original
    search target is used untouched, so the result is byte-identical to PR #200.
    """
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(paths["train_source_indexes"], allow_pickle=False)
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    batches = [
        _batch([rows[int(index)] for index in source], indexes, device)
        for indexes in plan
    ]

    model = _new_model(device)
    load_checkpoint_into_model(model, paths["initialization_checkpoint"])
    apply_trainable_scope(model, TRAINABLE_SCOPE)

    incumbent = _new_model(device)
    load_checkpoint_into_model(incumbent, paths["initialization_checkpoint"])
    for parameter in incumbent.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(manifest["optimizer"]["lr"])
    )
    saved = {0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)}
    model.train()
    for step, batch in enumerate(batches, 1):
        if beta != 0.0:
            p_inc = incumbent_policy_batch(incumbent, batch)
            p_beta = mixed_policy_target(batch["p"], p_inc, batch["mask"], beta)
            assert_legal_distribution(
                p_beta.detach().cpu().numpy(),
                batch["mask"].detach().cpu().numpy(),
            )
            batch = {**batch, "p": p_beta}
        policy, value = _losses(model, batch)
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
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
    return saved


def policy_drift_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Legal-policy drift of ``candidate`` versus ``reference`` (per-state).

    Reports mean legal L1, mean legal JS, top-1 change rate, maximum per-state
    legal L1, and p50/p90/p95/p99 legal L1. Both inputs are full move vectors
    (zero on illegal moves, normalized over legal moves).
    """
    legal = mask.astype(bool)
    diffs = np.sum(np.abs(candidate - reference), axis=1)
    js_vals = js(candidate, reference)
    cand_top = np.argmax(np.where(legal, candidate, -np.inf), axis=1)
    ref_top = np.argmax(np.where(legal, reference, -np.inf), axis=1)
    return {
        "legal_l1_mean": float(np.mean(diffs)),
        "legal_l1_max": float(np.max(diffs)),
        "legal_l1_p50": float(np.quantile(diffs, 0.50)),
        "legal_l1_p90": float(np.quantile(diffs, 0.90)),
        "legal_l1_p95": float(np.quantile(diffs, 0.95)),
        "legal_l1_p99": float(np.quantile(diffs, 0.99)),
        "legal_js_mean": float(np.mean(js_vals)),
        "top1_change_rate": float(np.mean(cand_top != ref_top)),
    }


def _cross_entropy(candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    return -np.sum(target * np.log(np.clip(candidate, 1e-12, None)), axis=1)


def probe_target_metrics(
    probe_rows: list[dict[str, Any]],
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    incumbent_state: dict[str, torch.Tensor],
    beta: float,
    beta000_step46_search_ce: float | None,
) -> dict[str, Any]:
    """CE and drift metrics for one lane across all checkpoints.

    Reported per checkpoint:
    - CE(candidate, search target)  -> p_search
    - CE(candidate, incumbent)      -> p_inc
    - CE(candidate, mixed target)   -> p_beta (this lane's beta)
    - search-target CE improvement vs incumbent
    - teacher-fit fraction (uses beta000_step46_search_ce as denominator)
    - policy drift vs incumbent (L1/JS/top1/max/percentiles)
    - policy drift vs search target
    """
    x = np.asarray([row["state"] for row in probe_rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    p_search = np.asarray([row["policy"] for row in probe_rows], dtype=np.float64)
    incumbent_policy, _incumbent_value = model_outputs(incumbent_state, x, mask)
    incumbent_policy = incumbent_policy.astype(np.float64)
    if beta == 0.0:
        mixed = p_search
    else:
        mixed = (1.0 - beta) * p_search + beta * incumbent_policy
        mixed = np.where(mask.astype(bool), mixed, 0.0)
        totals = mixed.sum(axis=1, keepdims=True)
        bad = ~np.isfinite(totals) | (totals <= 0.0)
        uniform = np.where(mask, 1.0, 0.0)
        uniform = uniform / uniform.sum(axis=1, keepdims=True)
        mixed = np.where(bad, uniform, mixed / totals)
    mixed = mixed.astype(np.float64)

    incumbent_search_ce = float(np.mean(_cross_entropy(incumbent_policy, p_search)))
    result: dict[str, Any] = {}
    for step, (state, _optimizer) in snapshots.items():
        candidate, _value = model_outputs(state, x, mask)
        candidate = candidate.astype(np.float64)
        ce_search = float(np.mean(_cross_entropy(candidate, p_search)))
        ce_inc = float(np.mean(_cross_entropy(candidate, incumbent_policy)))
        ce_mixed = float(np.mean(_cross_entropy(candidate, mixed)))
        improvement = incumbent_search_ce - ce_search
        fit_fraction: float | None
        if beta000_step46_search_ce is None:
            fit_fraction = None
        else:
            denom = incumbent_search_ce - beta000_step46_search_ce
            if denom <= 1e-12:
                fit_fraction = None
            else:
                fit_fraction = float(improvement / denom)
        result[str(step)] = {
            "ce_candidate_search": ce_search,
            "ce_candidate_incumbent": ce_inc,
            "ce_candidate_mixed": ce_mixed,
            "ce_incumbent_search": incumbent_search_ce,
            "search_target_ce_improvement_vs_incumbent": improvement,
            "fit_fraction": fit_fraction,
            "fit_denominator": (
                incumbent_search_ce - beta000_step46_search_ce
                if beta000_step46_search_ce is not None
                else None
            ),
            "drift_vs_incumbent": policy_drift_metrics(
                candidate, incumbent_policy, mask
            ),
            "drift_vs_search_target": policy_drift_metrics(candidate, p_search, mask),
        }
    return result


def _win_draw_loss(records: list[dict[str, Any]]) -> dict[str, int]:
    wins = draws = losses = 0
    for record in records:
        score = score_from_game(record)
        if score > 0.5:
            wins += 1
        elif score < 0.5:
            losses += 1
        else:
            draws += 1
    return {"wins": wins, "draws": draws, "losses": losses}


def paired_arena(
    artifacts: dict[str, dict[int, Path]],
    current: Path,
    workdir: Path,
    workers: int,
    targets: list[tuple[str, int, str]],
) -> dict[str, Any]:
    """Canonical paired arena for explicit (lane, step, context) targets.

    Reuses the cached ``_arena_records`` machinery (128 openings, seat swap,
    seed contract, frozen incumbent opponent, bootstrap CI). Reports paired
    effect, 95% CI, P0/P1, and W/D/L from the candidate (challenger) records.
    """
    metrics: dict[str, Any] = {}
    for lane, step, context in targets:
        control = _arena_records(
            workdir, current, current, context, "current_control", workers
        )
        candidate_records = _arena_records(
            workdir / f"{lane}_step_{step:04d}",
            artifacts[lane][step],
            current,
            context,
            f"{lane}_vs_current",
            workers,
        )
        effect = paired_opening_candidate_effect(candidate_records, control)
        metrics.setdefault(lane, {}).setdefault(str(step), {})[context] = {
            "paired_candidate_effect": effect["paired_candidate_effect"],
            "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
            "p0_effect": effect["p0_effect"],
            "p1_effect": effect["p1_effect"],
            "win_draw_loss": _win_draw_loss(candidate_records),
            "orientation": "candidate_minus_incumbent",
        }
    return metrics


def arena_safe(entry: dict[str, Any]) -> bool:
    """Prespecified noninferiority gate: CI includes zero OR lower >= -0.03."""
    ci = entry["opening_bootstrap_ci"]
    return ci["upper_95"] >= 0.0 or ci["lower_95"] >= NONINFERIORITY_LOWER


def _ci_excludes_zero(entry: dict[str, Any]) -> bool:
    return entry["opening_bootstrap_ci"]["upper_95"] < 0.0


NEXT_EXPERIMENTS = {
    "safe_learning_window_found": (
        "test the winning beta in one normal AlphaZero-style iteration with fresh "
        "self-play (NOT implemented in this PR)"
    ),
    "anchor_only_freezes_learning": (
        "do not increase generic anchor strength; inspect which moves/states "
        "produce harmful prior redistribution"
    ),
    "same_state_anchor_insufficient": (
        "replace global divergence control with a selective target construction "
        "experiment based on search-sensitive states/actions"
    ),
    "non_monotonic_anchor_response": (
        "run a per-state/action causal sensitivity analysis rather than another "
        "scalar beta sweep"
    ),
    "invariant_failure": (
        "fix the invariant (beta_000 PR #200 reproduction or beta_100 incumbent "
        "equivalence) before interpreting anchor results"
    ),
    "inconclusive": (
        "confidence intervals do not distinguish the relevant cases; rerun with "
        "more openings or seeds"
    ),
}


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the prespecified same-state-anchor decision rule."""
    reproduction = summary.get("deterministic_reproduction", {})
    beta000_repro = bool(reproduction.get("pr200_policy_head_state_hashes"))
    beta100_equiv = bool(summary.get("sanity", {}).get("beta_100_incumbent_equivalent"))
    if not beta000_repro or not beta100_equiv:
        label = "invariant_failure"
        evidence: dict[str, Any] = {
            "beta_000_reproduces_pr200": beta000_repro,
            "beta_100_incumbent_equivalent": beta100_equiv,
        }
        return {
            "label": label,
            "next_experiment": NEXT_EXPERIMENTS[label],
            "evidence": evidence,
        }

    arena = summary.get("arena") or {}
    probe = summary.get("probe_target_metrics") or {}
    step46 = "46"

    def entry46(lane: str, context: str) -> dict[str, Any] | None:
        return arena.get(lane, {}).get(step46, {}).get(context)

    def fit46(lane: str) -> float | None:
        return probe.get(lane, {}).get(step46, {}).get("fit_fraction")

    beta_lt1 = [name for name, beta in BETA_LANES if beta < 1.0]
    # Genuinely-anchored lanes (0 < beta < 1): beta_000 is the unanchored
    # baseline (definitionally fit_fraction = 1.0 and, since it reproduces PR
    # #200, materially negative) and beta_100 is the incumbent-equivalence
    # reference. The anchor analysis concerns the intermediate lanes only.
    beta_anchor = [name for name, beta in BETA_LANES if 0.0 < beta < 1.0]
    beta_order = [name for name, _ in BETA_LANES]

    safe_lanes_lt1 = [
        lane
        for lane in beta_lt1
        if (e := entry46(lane, STAGE1_CONTEXT)) is not None and arena_safe(e)
    ]
    meaningful_fit_lanes_lt1 = [
        lane
        for lane in beta_lt1
        if (fit := fit46(lane)) is not None and fit >= MEANINGFUL_FIT_FRACTION
    ]
    safe_and_fit = [
        lane
        for lane in beta_lt1
        if (e := entry46(lane, STAGE1_CONTEXT)) is not None
        and arena_safe(e)
        and (fit := fit46(lane)) is not None
        and fit >= MEANINGFUL_FIT_FRACTION
    ]

    # High-budget gate for safe+fit lanes.
    high_budget_regression: dict[str, bool] = {}
    for lane in safe_and_fit:
        hi = entry46(lane, STAGE2_CONTEXT)
        if hi is None:
            high_budget_regression[lane] = False
        else:
            high_budget_regression[lane] = (
                _ci_excludes_zero(hi)
                and hi["opening_bootstrap_ci"]["lower_95"] < NONINFERIORITY_LOWER
            )

    safe_window = [
        lane for lane in safe_and_fit if not high_budget_regression.get(lane, False)
    ]

    # Monotonicity: step-46 384:256 paired effect should be roughly
    # non-decreasing (toward zero / less negative) as beta -> 1. A
    # non-monotonic response means a scalar beta sweep is the wrong axis
    # (particular probability redistributions matter more than global
    # divergence magnitude), so it is surfaced ahead of the generic
    # same-state-insufficiency diagnosis.
    effects: list[float | None] = []
    for lane in beta_order:
        e = entry46(lane, STAGE1_CONTEXT)
        effects.append(float(e["paired_candidate_effect"]) if e else None)
    non_monotonic = False
    last: float | None = None
    for value in effects:
        if value is None:
            continue
        if last is not None and value < last - 0.01:
            non_monotonic = True
            break
        last = value

    if safe_window:
        label = "safe_learning_window_found"
    elif non_monotonic:
        label = "non_monotonic_anchor_response"
    elif any(
        lane not in safe_lanes_lt1
        for lane in beta_anchor
        if (fit := fit46(lane)) is not None and fit >= MEANINGFUL_FIT_FRACTION
    ):
        # A genuinely-anchored (0 < beta < 1) lane retains meaningful
        # teacher fitting yet remains materially negative.
        label = "same_state_anchor_insufficient"
    elif (
        safe_lanes_lt1
        and all(lane == "beta_095" for lane in safe_lanes_lt1)
        and not any(
            (fit46(lane) or 0.0) >= MEANINGFUL_FIT_FRACTION for lane in beta_anchor
        )
    ):
        # Anchoring only becomes arena-safe at the heaviest anchor, and
        # even there the teacher fit is frozen below the meaningful
        # threshold.
        label = "anchor_only_freezes_learning"
    else:
        label = "inconclusive"

    evidence = {
        "beta_000_reproduces_pr200": beta000_repro,
        "beta_100_incumbent_equivalent": beta100_equiv,
        "step46_384_256_effects": {
            lane: (
                float(entry46(lane, STAGE1_CONTEXT)["paired_candidate_effect"])
                if entry46(lane, STAGE1_CONTEXT)
                else None
            )
            for lane, _ in BETA_LANES
        },
        "step46_fit_fractions": {lane: fit46(lane) for lane, _ in BETA_LANES},
        "arena_safe_lanes_lt1": safe_lanes_lt1,
        "step46_384_256_monotonic": not non_monotonic,
        "meaningful_teacher_fit_lanes_anchored": [
            lane
            for lane in beta_anchor
            if (fit := fit46(lane)) is not None and fit >= MEANINGFUL_FIT_FRACTION
        ],
        "meaningful_teacher_fit_lanes_lt1": meaningful_fit_lanes_lt1,
        "safe_and_fit_lanes": safe_and_fit,
        "high_budget_regression": high_budget_regression,
        "safe_window_lanes": safe_window,
        "noninferiority_lower": NONINFERIORITY_LOWER,
        "meaningful_fit_fraction": MEANINGFUL_FIT_FRACTION,
    }
    return {
        "label": label,
        "next_experiment": NEXT_EXPERIMENTS[label],
        "evidence": evidence,
    }


def markdown(summary: dict[str, Any]) -> str:
    """Render a compact committed record; complete detail remains in JSON."""
    classification = summary["classification"]
    label = classification["label"]
    lanes = [name for name, _ in BETA_LANES]
    lines = [
        "# AlphaZero-Lite Frozen-Trunk Same-State Anchor Results",
        "",
        f"**Classification:** `{label}`",
        "",
        f"- beta_000 reproduces PR #200 policy_head hashes: "
        f"`{summary['deterministic_reproduction']['pr200_policy_head_state_hashes']}`",
        f"- beta_100 incumbent-equivalent: "
        f"`{summary['sanity']['beta_100_incumbent_equivalent']}`",
        f"- all lanes start from identical incumbent: "
        f"`{summary['sanity']['lanes_start_identical']}`",
        f"- trunk byte-identical to incumbent (all lanes): "
        f"`{summary['sanity']['all_lanes_trunk_zero_change']}`",
        f"- value stack byte-identical to incumbent (all lanes): "
        f"`{summary['sanity']['all_lanes_value_stack_zero_change']}`",
        f"- checkpoint steps: `{summary['checkpoint_steps']}`",
        f"- current weights sha256: `{summary['inputs']['current_weights_sha256']}`",
        f"- replay sha256: `{summary['inputs']['replay_sha256']}`",
        f"- initialization checkpoint sha256: "
        f"`{summary['inputs']['initialization_checkpoint_sha256']}`",
        f"- seed: `{summary['inputs']['seed']}`",
        f"- optimizer: `{json.dumps(summary['inputs']['optimizer'])}`",
        f"- gradient clip: `{summary['inputs']['gradient_clip']}`",
        f"- trainable scope: `{TRAINABLE_SCOPE}`",
        "",
        "## Target construction",
        "",
        "For every training replay state x: "
        "`p_beta(x) = (1-beta) p_search(x) + beta p_inc(x)` "
        "with `p_inc(x)` the frozen incumbent legal-masked policy on that exact "
        "state (masking identical to the policy loss). Illegal mass zeroed and "
        "renormalized over legal moves. beta_000 uses the unmodified search "
        "target (byte-identical to PR #200).",
        "",
        "## Training / target metrics (frozen validation probe, step 46)",
        "",
        "| Lane | CE(search) | CE(incumbent) | CE(mixed) | search-CE improv | fit_fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    probe = summary.get("probe_target_metrics") or {}
    for lane in lanes:
        entry = probe.get(lane, {}).get("46")
        if entry is None:
            continue
        fit = entry["fit_fraction"]
        fit_str = "n/a" if fit is None else f"{fit:.4f}"
        lines.append(
            f"| {lane} | {entry['ce_candidate_search']:.4f} | "
            f"{entry['ce_candidate_incumbent']:.4f} | "
            f"{entry['ce_candidate_mixed']:.4f} | "
            f"{entry['search_target_ce_improvement_vs_incumbent']:+.4f} | "
            f"{fit_str} |"
        )
    lines.append(
        "fit_fraction = (search_CE_inc - search_CE_candidate) / "
        "(search_CE_inc - search_CE_beta000_step46). Reported `n/a` if the "
        "denominator is <= 0."
    )
    lines.extend(
        [
            "",
            "## Policy drift vs incumbent (frozen validation probe, step 46)",
            "",
            "| Lane | L1 mean | L1 max | L1 p50 | L1 p90 | L1 p95 | L1 p99 | "
            "JS mean | top-1 change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane in lanes:
        entry = probe.get(lane, {}).get("46")
        if entry is None:
            continue
        d = entry["drift_vs_incumbent"]
        lines.append(
            f"| {lane} | {d['legal_l1_mean']:.6f} | {d['legal_l1_max']:.6f} | "
            f"{d['legal_l1_p50']:.6f} | {d['legal_l1_p90']:.6f} | "
            f"{d['legal_l1_p95']:.6f} | {d['legal_l1_p99']:.6f} | "
            f"{d['legal_js_mean']:.6f} | {d['top1_change_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Policy drift vs search target (frozen validation probe, step 46)",
            "",
            "| Lane | L1 mean | L1 max | JS mean | top-1 change |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane in lanes:
        entry = probe.get(lane, {}).get("46")
        if entry is None:
            continue
        d = entry["drift_vs_search_target"]
        lines.append(
            f"| {lane} | {d['legal_l1_mean']:.6f} | {d['legal_l1_max']:.6f} | "
            f"{d['legal_js_mean']:.6f} | {d['top1_change_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"## Search diagnostics ({PUCT_CONTEXT} context, vs incumbent)",
            "",
            "| Lane | Step | Move change | Visit JS | Q-rank change | Root-value delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    puct = summary.get("puct") or {}
    for lane in lanes:
        metrics = puct.get(lane, {}).get("metrics", {})
        for step in summary["checkpoint_steps"]:
            entry = metrics.get(str(step), {}).get(PUCT_CONTEXT)
            if entry is None:
                continue
            lines.append(
                f"| {lane} | {step} | {entry['selected_move_change_rate']:.4f} | "
                f"{entry['visit_js']:.4f} | {entry['child_q_rank_change']:+.4f} | "
                f"{entry['root_value_delta']:+.4f} |"
            )
    per_depth = summary.get("per_depth_probe") or {}
    if per_depth:
        lines.extend(
            [
                "",
                "## Per-depth policy L1/JS on expanded probe states "
                "(step 46, 384:256, candidate vs incumbent)",
                "",
                "| Lane | Depth | Expanded | L1 mean | JS mean |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for lane in lanes:
            tel = (
                per_depth.get(lane, {})
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
                    f"| {lane} | {depth} | {b['expanded_nodes']} | "
                    f"{b['mean_pairwise_legal_l1']:.6f} | "
                    f"{b['mean_pairwise_legal_js']:.6f} |"
                )
    lines.extend(
        [
            "",
            "## Canonical arena (candidate versus frozen incumbent)",
            "",
            "| Lane | Step | Context | Paired effect | 95% CI | P0 | P1 | W/D/L |",
            "| --- | ---: | --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    arena = summary.get("arena") or {}
    for lane in lanes:
        for step in summary["checkpoint_steps"]:
            for context in (STAGE1_CONTEXT, STAGE2_CONTEXT):
                entry = arena.get(lane, {}).get(str(step), {}).get(context)
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
            "## Classification evidence",
            "",
            "| Signal | Value |",
            "| --- | ---: |",
        ]
    )
    evidence = classification["evidence"]
    for key, value in evidence.items():
        if isinstance(value, bool):
            rendered = str(value)
        elif isinstance(value, dict):

            def _fmt(v: Any) -> Any:
                if v is None:
                    return None
                if isinstance(v, bool):
                    return bool(v)
                if isinstance(v, (int, float)):
                    return f"{v:+.4f}"
                return v

            rendered = json.dumps({k: _fmt(v) for k, v in value.items()})
        elif isinstance(value, list):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.extend(
        [
            "",
            "## Recommended next experiment (not implemented here)",
            "",
            f"`{classification['next_experiment']}`",
            "",
            "## Exact commands",
            "",
            "```bash",
            "python ml/alphazero_lite/run_frozen_trunk_same_state_anchor_ablation.py \\",
            "  --pr191-workdir /tmp/azlite_shared_trunk_learning \\",
            "  --workdir /tmp/azlite_frozen_trunk_same_state_anchor \\",
            "  --arena-workers 24",
            "```",
            "",
            "Full evidence: "
            "`docs/data/alphazero-lite-frozen-trunk-same-state-anchor-summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr191-workdir", type=Path, default=Path("/tmp/azlite_shared_trunk_learning")
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_frozen_trunk_same_state_anchor"),
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-frozen-trunk-same-state-anchor-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-frozen-trunk-same-state-anchor-results.md",
    )
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--per-depth-probe", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--stage2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the 1200:1200 high-budget gate for arena-safe+fit lanes",
    )
    parser.add_argument("--arena-workers", type=int, default=24)
    args = parser.parse_args()

    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact does not match the PR191 initialization")
    if manifest["architecture"]["model_type"] != "residual_v3":
        raise RuntimeError(
            "same-state anchor lanes require the residual_v3 architecture"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(manifest["seed"])
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    batch_count = int(len(plan))
    steps = checkpoint_steps(batch_count)

    lanes: dict[str, dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]] = {}
    betas: dict[str, float] = {}
    for lane, beta in BETA_LANES:
        configure_determinism(device, seed)
        lanes[lane] = replay_lane_anchored(
            manifest, args.workdir / lane, device, beta, steps
        )
        betas[lane] = beta

    incumbent = lanes[BETA_LANES[0][0]][0][0]
    lanes_start_identical = all(
        tensors_identical(lanes[lane][0][0], incumbent) for lane, _ in BETA_LANES
    )
    if not lanes_start_identical:
        raise RuntimeError("anchor lanes do not start from identical parameters")

    all_lanes_trunk_zero = all(
        trunk_parameters_identical(state, incumbent)
        for lane, _ in BETA_LANES
        for state, _optimizer in lanes[lane].values()
    )
    all_lanes_value_stack_zero = all(
        group_parameters_identical(state, incumbent, VALUE_STACK_PREFIXES)
        for lane, _ in BETA_LANES
        for state, _optimizer in lanes[lane].values()
    )
    if not all_lanes_trunk_zero:
        raise RuntimeError("an anchor lane changed a trunk parameter")
    if not all_lanes_value_stack_zero:
        raise RuntimeError("an anchor lane changed a value-stack parameter")

    beta000_hashes = state_hashes(lanes["beta_000"])
    beta000_repro = beta000_hashes == PR200_POLICY_HEAD_STATE_HASHES
    if not beta000_repro:
        raise RuntimeError("beta_000 does not reproduce the PR #200 policy_head hashes")

    # beta_100 incumbent-equivalence: policy-head drift below tolerance AND
    # search-equivalence (verified later via arena/probe; recorded here from drift).
    beta100_drift = group_delta(lanes["beta_100"][46][0], incumbent)["policy_head"]
    beta100_incumbent_equivalent = beta100_drift < BETA100_DRIFT_TOL

    rows = read_jsonl(Path(manifest["replay_path"]))
    validation_indexes = np.load(paths["validation_source_indexes"], allow_pickle=False)
    probe, probe_manifest = decoded_validation_manifest(rows, validation_indexes)
    probe_manifest["validation_source_indexes_sha256"] = sha256_file(
        paths["validation_source_indexes"]
    )
    probe_manifest["replay_sha256"] = sha256_file(Path(manifest["replay_path"]))
    probe_manifest["manifest_sha256"] = stable_hash(probe_manifest)
    probe_rows = [rows[index] for index in probe_manifest["source_indexes"]]

    # beta_000 step-46 search CE is the teacher-fit denominator reference.
    beta000_probe = probe_target_metrics(
        probe_rows, lanes["beta_000"], incumbent, 0.0, None
    )
    beta000_step46_search_ce = beta000_probe["46"]["ce_candidate_search"]

    probe_metrics: dict[str, Any] = {}
    for lane, beta in BETA_LANES:
        probe_metrics[lane] = probe_target_metrics(
            probe_rows, lanes[lane], incumbent, beta, beta000_step46_search_ce
        )

    # Output drift (CE vs search target + value huber) reused from PR #200.
    output: dict[str, Any] = {}
    drift: dict[str, Any] = {}
    probe_drift: dict[str, Any] = {}
    for lane, _beta in BETA_LANES:
        output[lane] = with_total_loss(output_drift(probe_rows, lanes[lane]))
        probe_drift[lane] = probe_output_drift(probe_rows, lanes[lane])
        drift[lane] = {
            str(step): group_delta(state, incumbent)
            for step, (state, _optimizer) in lanes[lane].items()
        }
    output["incumbent"] = with_total_loss(
        output_drift(probe_rows, {0: (incumbent, {})})
    )
    probe_drift["incumbent"] = probe_output_drift(probe_rows, {0: (incumbent, {})})
    drift["incumbent"] = {"0": group_delta(incumbent, incumbent)}

    artifacts = {
        lane: export_snapshot_artifacts(snapshots, args.workdir / lane)
        for lane, snapshots in lanes.items()
    }
    incumbent_artifacts = export_snapshot_artifacts(
        {0: (incumbent, {})}, args.workdir / "incumbent"
    )

    puct: dict[str, Any] = {}
    if args.puct:
        probe_hash = stable_hash(probe_manifest)
        for lane, _beta in BETA_LANES:
            puct[lane] = puct_trajectory(
                probe[:PROBE_SIZE],
                artifacts[lane],
                args.workdir / lane,
                probe_hash,
                contexts=(PUCT_CONTEXT,),
            )

    arena: dict[str, Any] = {}
    if args.arena:
        stage1_targets = [
            (lane, step, STAGE1_CONTEXT)
            for lane, _beta in BETA_LANES
            for step in STAGE1_STEPS
            if step in artifacts[lane]
        ]
        arena = paired_arena(
            artifacts,
            args.current,
            args.workdir / "arena",
            args.arena_workers,
            stage1_targets,
        )

    # Stage 2 (high-budget gate) only for arena-safe AND meaningful-fit lanes.
    if args.arena and args.stage2:
        stage2_targets: list[tuple[str, int, str]] = []
        for lane, beta in BETA_LANES:
            if beta >= 1.0:
                continue
            entry = arena.get(lane, {}).get(str(STAGE2_STEP), {}).get(STAGE1_CONTEXT)
            fit = (
                probe_metrics.get(lane, {})
                .get(str(STAGE2_STEP), {})
                .get("fit_fraction")
            )
            if entry is None or fit is None:
                continue
            if arena_safe(entry) and fit >= MEANINGFUL_FIT_FRACTION:
                stage2_targets.append((lane, STAGE2_STEP, STAGE2_CONTEXT))
        if stage2_targets:
            stage2 = paired_arena(
                artifacts,
                args.current,
                args.workdir / "arena",
                args.arena_workers,
                stage2_targets,
            )
            for lane, step, context in stage2_targets:
                arena.setdefault(lane, {}).setdefault(str(step), {})[context] = stage2[
                    lane
                ][str(step)][context]

    per_depth: dict[str, Any] = {}
    if args.per_depth_probe:
        for lane, _beta in BETA_LANES:
            candidate_path = artifacts[lane][STAGE2_STEP]
            per_depth[lane] = puct_probe(
                probe,
                candidate_path,
                incumbent_artifacts[0],
                PUCT_CONTEXT,
                modes=("incumbent_all",),
            )

    summary: dict[str, Any] = {
        "schema": NAMESPACE,
        "guardrails": {
            "promotion": False,
            "new_self_play": False,
            "target_change": "policy_only_same_state_mixture",
            "value_target_change": False,
            "lr_change": False,
            "loss_weight_change": False,
            "architecture_change": False,
            "search_change": False,
            "diagnostic_only": True,
        },
        "inputs": {
            "current_weights_sha256": CURRENT_HASH,
            "replay_sha256": sha256_file(Path(manifest["replay_path"])),
            "training_manifest_sha256": sha256_file(
                args.pr191_workdir / "training_manifest.json"
            ),
            "initialization_checkpoint_sha256": sha256_file(
                paths["initialization_checkpoint"]
            ),
            "seed": seed,
            "optimizer": manifest["optimizer"],
            "gradient_clip": manifest["gradient_clip"],
            "value_loss": manifest["value_loss"],
            "value_loss_weight": manifest["value_loss_weight"],
            "huber_delta": manifest["huber_delta"],
            "policy_loss": manifest["policy_loss"],
            "architecture": manifest["architecture"],
            "batch_plan": manifest["batch_plan"],
            "batch_count": batch_count,
            "trainable_scope": TRAINABLE_SCOPE,
            "beta_lanes": [{"lane": lane, "beta": beta} for lane, beta in BETA_LANES],
            "lane_trainable_scope": {
                lane: lane_trainable_scope(TRAINABLE_SCOPE) for lane, _ in BETA_LANES
            },
            "optimizer_step_counts": {
                lane: list(snapshots.keys()) for lane, snapshots in lanes.items()
            },
            "evaluation": {
                "opponent": "frozen incumbent (model-artifact/current)",
                "openings": "canonical 128-opening suite, 2 games per opening per seat",
                "stage1_context": STAGE1_CONTEXT,
                "stage1_steps": list(STAGE1_STEPS),
                "stage2_context": STAGE2_CONTEXT,
                "stage2_step": STAGE2_STEP,
                "puct_context": PUCT_CONTEXT,
            },
        },
        "deterministic_reproduction": {
            "pr200_policy_head_state_hashes": beta000_repro,
        },
        "sanity": {
            "lanes_start_identical": lanes_start_identical,
            "all_lanes_trunk_zero_change": all_lanes_trunk_zero,
            "all_lanes_value_stack_zero_change": all_lanes_value_stack_zero,
            "beta_100_incumbent_equivalent": beta100_incumbent_equivalent,
            "beta_100_policy_head_drift": beta100_drift,
            "beta_100_drift_tolerance": BETA100_DRIFT_TOL,
        },
        "checkpoint_steps": steps,
        "state_hashes": {
            lane: state_hashes(snapshots) for lane, snapshots in lanes.items()
        },
        "drift": drift,
        "output_drift": output,
        "probe_output_drift": probe_drift,
        "probe_target_metrics": probe_metrics,
        "probe_manifest": probe_manifest,
        "puct": puct,
        "per_depth_probe": per_depth,
        "arena": arena,
    }
    summary["classification"] = classify(summary)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary["classification"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
