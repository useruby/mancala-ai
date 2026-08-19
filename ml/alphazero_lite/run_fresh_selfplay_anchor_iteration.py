#!/usr/bin/env python3
# ruff: noqa: E402
"""Fresh self-play same-state incumbent-anchored policy-target iteration.

PR #202 identified that a same-state incumbent anchor:
    p_beta(x) = (1 - beta) * p_search(x) + beta * p_inc(x)
at beta = 0.95 rescued the frozen-trunk policy-head update from the severe
low-budget arena regression (effect -0.1895 -> -0.0137 @ 384:256, +0.0039 @
1200:1200) while retaining 47.15% teacher-fitting fraction on historical
PR #199/#200 replay data.

This experiment tests whether beta_095 remains a safe, meaningful policy
update when training on fresh self-play generated from the frozen incumbent
(AlphaZero iteration transfer test), without tuning beta.

Protocol:
1. Generate fresh self-play exactly once using the frozen incumbent.
2. Run dataset-shift diagnostics comparing fresh replay to historical PR #202
   replay.
3. Run matched training lanes from the identical incumbent checkpoint:
   - incumbent (no updates)
   - beta_000 (pure fresh search target)
   - beta_095 (mixed fresh search + same-state incumbent anchor)
   - beta_100 (pure same-state incumbent target)
4. Verify invariants (zero trunk drift, zero value drift, beta_100 equivalence).
5. Evaluate training metrics, search diagnostics (PUCT probe), and canonical
   paired arena (384:256 and 1200:1200 high-budget gate).
6. Classify according to the prespecified decision rule.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
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
from ml.alphazero_lite.run_frozen_trunk_distillation_ablation import (  # noqa: E402
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
from ml.alphazero_lite.run_frozen_trunk_same_state_anchor_ablation import (  # noqa: E402
    BETA100_DRIFT_TOL,
    MEANINGFUL_FIT_FRACTION,
    NONINFERIORITY_LOWER,
    _cross_entropy,
    _win_draw_loss,
    arena_safe,
    assert_legal_distribution,
    incumbent_policy_batch,
    mixed_policy_target,
    policy_drift_metrics,
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

NAMESPACE = "azlite_fresh_selfplay_anchor_v1"
BETA_LANES: tuple[tuple[str, float], ...] = (
    ("beta_000", 0.00),
    ("beta_095", 0.95),
    ("beta_100", 1.00),
)
TRAINABLE_SCOPE = "policy_head"
CHECKPOINT_STEPS = [1, 4, 16, 46]
PRIMARY_CONTEXT = "384:256"
PUCT_CONTEXT = "384:256"
STAGE1_CONTEXT = "384:256"
STAGE2_CONTEXT = "1200:1200"
STAGE1_STEPS = (16, 46)
STAGE2_STEP = 46
PROBE_SIZE = 256

DEFAULT_GAMES = 700
DEFAULT_SEED = 42
DEFAULT_SIMULATIONS = 384
DEFAULT_C_PUCT = 1.25
DEFAULT_BATCH_SIZE = 512
DEFAULT_LR = 1e-5
DEFAULT_GRAD_CLIP = 1.0

NEXT_EXPERIMENTS = {
    "fresh_safe_window_replicated": (
        "test the winning beta in a second AlphaZero-style iteration: "
        "promote beta_095 candidate inside experiment workspace, generate new "
        "self-play from it, train with same-state beta_095 anchoring to parent, "
        "and arena candidate vs parent and original incumbent (NOT implemented in this PR)"
    ),
    "fixed_replay_window_did_not_generalize": (
        "stop scalar-beta tuning and identify the high-disagreement fresh "
        "states/actions responsible for the regression"
    ),
    "fresh_unanchored_update_safe": (
        "do not proceed immediately to multi-iteration beta_095 training; "
        "investigate replay-distribution differences between historical and fresh data"
    ),
    "anchor_suppresses_fresh_learning": (
        "investigate whether weaker anchor fractions or state-selective anchoring "
        "allow sufficient gradient signal while maintaining stability"
    ),
    "invariant_failure": (
        "fix the invariant failure before interpreting fresh-data results"
    ),
    "inconclusive": (
        "confidence intervals do not distinguish the relevant cases; rerun with "
        "more openings or seeds"
    ),
}


def python_executable() -> str:
    venv_python = REPO_ROOT / ".venv/bin/python"
    if venv_python.is_file() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    return sys.executable


def generate_fresh_self_play(
    *,
    out_path: Path,
    checkpoint_npz: Path,
    games: int,
    seed: int,
    simulations: int,
    c_puct: float,
    workers: int,
) -> dict[str, Any]:
    """Generate fresh self-play from incumbent checkpoint using canonical config."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_executable(),
        str(REPO_ROOT / "ml/alphazero_lite/self_play.py"),
        "--out",
        str(out_path),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint_npz),
        "--input-encoding",
        "kalah_v3",
        "--simulations",
        str(simulations),
        "--c-puct",
        str(c_puct),
        "--player-mode",
        "puct",
        "--root-policy-mode",
        "visit_count",
        "--policy-target-mode",
        "default",
        "--value-target-mode",
        "default",
        "--policy-target-noise-mode",
        "noisy",
        "--dirichlet-alpha",
        "0.3",
        "--dirichlet-epsilon",
        "0.3",
        "--tree-reuse-enabled",
        "--write-game-metadata",
        "--workers",
        str(workers),
    ]
    print(f"[self_play] generating {games} games with {workers} workers...", flush=True)
    res = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if res.returncode != 0:
        raise RuntimeError(f"self_play failed: {res.stderr[-2000:]}")

    replay_sha = sha256_file(out_path)
    rows = read_jsonl(out_path)
    game_indices = set(int(r["game_index"]) for r in rows if "game_index" in r)

    metadata = {
        "command": cmd,
        "games_requested": games,
        "games_generated": len(game_indices),
        "positions_generated": len(rows),
        "seed": seed,
        "simulations": simulations,
        "c_puct": c_puct,
        "player_mode": "puct",
        "input_encoding": "kalah_v3",
        "policy_target_mode": "default",
        "value_target_mode": "default",
        "policy_target_noise_mode": "noisy",
        "tree_reuse_enabled": True,
        "replay_path": str(out_path),
        "replay_sha256": replay_sha,
        "checkpoint_npz_sha256": sha256_file(checkpoint_npz),
    }
    return metadata


def _quantiles(values: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _entropy(p: np.ndarray) -> np.ndarray:
    clamped = np.clip(p, 1e-12, None)
    return -np.sum(p * np.log(clamped), axis=1)


def compute_dataset_diagnostics(
    rows: list[dict[str, Any]],
    incumbent_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compute distribution diagnostics on a replay dataset."""
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    p_search = np.asarray([row["policy"] for row in rows], dtype=np.float64)

    incumbent_policy, _inc_value = model_outputs(incumbent_state, x, mask)
    incumbent_policy = incumbent_policy.astype(np.float64)

    # Game lengths
    games_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        games_by_id[int(row["game_index"])].append(row)
    game_lengths = [len(g_rows) for g_rows in games_by_id.values()]

    # Player / seat distribution
    player_counts = Counter(int(row["player"]) for row in rows)
    total_positions = len(rows)

    # Legal move counts
    legal_counts = np.sum(mask > 0, axis=1)
    legal_move_dist = Counter(int(c) for c in legal_counts)

    # Entropies
    search_ent = _entropy(p_search)
    inc_ent = _entropy(incumbent_policy)

    # Divergence
    l1 = np.sum(np.abs(p_search - incumbent_policy), axis=1)
    js_div = js(p_search, incumbent_policy)

    cand_top = np.argmax(np.where(mask.astype(bool), p_search, -np.inf), axis=1)
    inc_top = np.argmax(np.where(mask.astype(bool), incumbent_policy, -np.inf), axis=1)
    top1_disagree = float(np.mean(cand_top != inc_top))

    # Values / outcomes
    values = [float(row.get("value", 0.0)) for row in rows]
    value_counts = Counter(round(v, 2) for v in values)

    return {
        "game_count": len(games_by_id),
        "position_count": total_positions,
        "game_length": _quantiles(game_lengths),
        "player_distribution": {
            "p0_count": player_counts[0],
            "p1_count": player_counts[1],
            "p0_fraction": player_counts[0] / max(1, total_positions),
            "p1_fraction": player_counts[1] / max(1, total_positions),
        },
        "legal_move_count_distribution": {
            str(k): {
                "count": count,
                "fraction": count / max(1, total_positions),
            }
            for k, count in sorted(legal_move_dist.items())
        },
        "search_policy_entropy": _quantiles(search_ent),
        "incumbent_policy_entropy": _quantiles(inc_ent),
        "legal_l1_search_vs_incumbent": _quantiles(l1),
        "legal_js_search_vs_incumbent": _quantiles(js_div),
        "top1_disagreement_rate": top1_disagree,
        "value_distribution": {
            str(k): {
                "count": count,
                "fraction": count / max(1, total_positions),
            }
            for k, count in sorted(value_counts.items())
        },
    }


def dataset_shift_comparison(
    fresh_rows: list[dict[str, Any]],
    historical_rows: list[dict[str, Any]],
    incumbent_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare fresh replay with historical PR #202 replay."""
    fresh_diag = compute_dataset_diagnostics(fresh_rows, incumbent_state)
    hist_diag = compute_dataset_diagnostics(historical_rows, incumbent_state)
    return {
        "fresh": fresh_diag,
        "historical": hist_diag,
        "deltas": {
            "position_count_delta": fresh_diag["position_count"]
            - hist_diag["position_count"],
            "game_count_delta": fresh_diag["game_count"] - hist_diag["game_count"],
            "search_entropy_mean_delta": (
                fresh_diag["search_policy_entropy"]["mean"]
                - hist_diag["search_policy_entropy"]["mean"]
            ),
            "incumbent_entropy_mean_delta": (
                fresh_diag["incumbent_policy_entropy"]["mean"]
                - hist_diag["incumbent_policy_entropy"]["mean"]
            ),
            "l1_mean_delta": (
                fresh_diag["legal_l1_search_vs_incumbent"]["mean"]
                - hist_diag["legal_l1_search_vs_incumbent"]["mean"]
            ),
            "js_mean_delta": (
                fresh_diag["legal_js_search_vs_incumbent"]["mean"]
                - hist_diag["legal_js_search_vs_incumbent"]["mean"]
            ),
            "top1_disagreement_delta": (
                fresh_diag["top1_disagreement_rate"]
                - hist_diag["top1_disagreement_rate"]
            ),
        },
    }


def replay_lane_fresh(
    manifest: dict[str, Any],
    workdir: Path,
    device: torch.device,
    beta: float,
    steps: list[int],
) -> tuple[dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]], float]:
    """Train one matched policy_head lane on fresh replay data."""
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

    initial_policy_grad_norm = 0.0
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


def probe_target_metrics_fresh(
    probe_rows: list[dict[str, Any]],
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    incumbent_state: dict[str, torch.Tensor],
    beta: float,
    beta000_step46_search_ce: float | None,
) -> dict[str, Any]:
    """CE, fit_fraction, and drift metrics on the fresh validation probe."""
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


def paired_arena(
    artifacts: dict[str, dict[int, Path]],
    current: Path,
    workdir: Path,
    workers: int,
    targets: list[tuple[str, int, str]],
) -> dict[str, Any]:
    """Canonical paired arena for explicit (lane, step, context) targets."""
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


def classify_fresh(summary: dict[str, Any]) -> dict[str, Any]:
    """Prespecified classification rule for fresh self-play iteration."""
    sanity = summary.get("sanity", {})
    lanes_start = bool(sanity.get("lanes_start_identical"))
    trunk_zero = bool(sanity.get("all_lanes_trunk_zero_change"))
    value_zero = bool(sanity.get("all_lanes_value_stack_zero_change"))
    beta100_equiv = bool(sanity.get("beta_100_incumbent_equivalent"))

    if not lanes_start or not trunk_zero or not value_zero or not beta100_equiv:
        label = "invariant_failure"
        return {
            "label": label,
            "next_experiment": NEXT_EXPERIMENTS[label],
            "evidence": {
                "lanes_start_identical": lanes_start,
                "all_lanes_trunk_zero_change": trunk_zero,
                "all_lanes_value_stack_zero_change": value_zero,
                "beta_100_incumbent_equivalent": beta100_equiv,
            },
        }

    arena = summary.get("arena") or {}
    probe = summary.get("probe_target_metrics") or {}
    step46 = "46"

    def entry(lane: str, step: str, context: str) -> dict[str, Any] | None:
        return arena.get(lane, {}).get(step, {}).get(context)

    def fit(lane: str, step: str) -> float | None:
        return probe.get(lane, {}).get(step, {}).get("fit_fraction")

    beta000_entry_384 = entry("beta_000", step46, STAGE1_CONTEXT)
    beta095_entry_384 = entry("beta_095", step46, STAGE1_CONTEXT)
    beta095_entry_1200 = entry("beta_095", step46, STAGE2_CONTEXT)
    beta095_fit = fit("beta_095", step46)

    beta000_safe = beta000_entry_384 is not None and arena_safe(beta000_entry_384)
    beta095_safe_384 = beta095_entry_384 is not None and arena_safe(beta095_entry_384)
    beta095_meaningful_fit = (
        beta095_fit is not None and beta095_fit >= MEANINGFUL_FIT_FRACTION
    )
    beta095_safe_1200 = beta095_entry_1200 is not None and arena_safe(
        beta095_entry_1200
    )

    evidence = {
        "beta_100_incumbent_equivalent": beta100_equiv,
        "beta_000_step46_384_256_effect": (
            float(beta000_entry_384["paired_candidate_effect"])
            if beta000_entry_384
            else None
        ),
        "beta_000_step46_384_256_safe": beta000_safe,
        "beta_095_step46_384_256_effect": (
            float(beta095_entry_384["paired_candidate_effect"])
            if beta095_entry_384
            else None
        ),
        "beta_095_step46_384_256_safe": beta095_safe_384,
        "beta_095_step46_fit_fraction": beta095_fit,
        "beta_095_step46_meaningful_fit": beta095_meaningful_fit,
        "beta_095_step46_1200_1200_effect": (
            float(beta095_entry_1200["paired_candidate_effect"])
            if beta095_entry_1200
            else None
        ),
        "beta_095_step46_1200_1200_safe": beta095_safe_1200,
        "noninferiority_lower": NONINFERIORITY_LOWER,
        "meaningful_fit_fraction": MEANINGFUL_FIT_FRACTION,
    }

    if beta000_safe:
        label = "fresh_unanchored_update_safe"
    elif beta095_safe_384 and beta095_meaningful_fit and beta095_safe_1200:
        label = "fresh_safe_window_replicated"
    elif beta095_meaningful_fit and not beta095_safe_384:
        label = "fixed_replay_window_did_not_generalize"
    elif beta095_safe_384 and not beta095_meaningful_fit:
        label = "anchor_suppresses_fresh_learning"
    else:
        label = "inconclusive"

    return {
        "label": label,
        "next_experiment": NEXT_EXPERIMENTS[label],
        "evidence": evidence,
    }


def markdown_report(summary: dict[str, Any]) -> str:
    """Render comprehensive markdown report matching repo conventions."""
    classification = summary["classification"]
    label = classification["label"]
    lanes = [name for name, _ in BETA_LANES]
    inputs = summary["inputs"]
    shift = summary.get("dataset_shift_diagnostics", {})
    fresh_d = shift.get("fresh", {})
    hist_d = shift.get("historical", {})

    lines = [
        "# AlphaZero-Lite Fresh Self-Play Anchor Validation Results",
        "",
        f"**Classification:** `{label}`",
        "",
        f"**Recommended Next Action:** `{classification['next_experiment']}`",
        "",
        "## Core Guardrails & Invariants",
        "",
        f"- all lanes start from identical incumbent: "
        f"`{summary['sanity']['lanes_start_identical']}`",
        f"- trunk byte-identical to incumbent (all lanes): "
        f"`{summary['sanity']['all_lanes_trunk_zero_change']}`",
        f"- value stack byte-identical to incumbent (all lanes): "
        f"`{summary['sanity']['all_lanes_value_stack_zero_change']}`",
        f"- beta_100 incumbent-equivalent: "
        f"`{summary['sanity']['beta_100_incumbent_equivalent']}` "
        f"(drift = {summary['sanity']['beta_100_policy_head_drift']:.2e}, tol = {summary['sanity']['beta_100_drift_tolerance']})",
        f"- beta_100 initial policy gradient norm: "
        f"`{summary['sanity']['beta_100_initial_policy_grad_norm']:.2e}`",
        f"- checkpoint steps: `{summary['checkpoint_steps']}`",
        f"- incumbent weights sha256: `{inputs['current_weights_sha256']}`",
        f"- fresh replay sha256: `{inputs['fresh_selfplay']['replay_sha256']}`",
        f"- seed: `{inputs['seed']}`",
        f"- optimizer: `{json.dumps(inputs['optimizer'])}`",
        f"- gradient clip: `{inputs['gradient_clip']}`",
        f"- trainable scope: `{TRAINABLE_SCOPE}`",
        "",
        "## Dataset-Shift Diagnostics (Fresh Replay vs PR #202 Historical Replay)",
        "",
        "| Metric | Fresh Replay | PR #202 Historical Replay | Delta (Fresh - Hist) |",
        "| --- | ---: | ---: | ---: |",
    ]

    if fresh_d and hist_d:
        lines.extend(
            [
                f"| Games | {fresh_d.get('game_count', 0)} | {hist_d.get('game_count', 0)} | "
                f"{fresh_d.get('game_count', 0) - hist_d.get('game_count', 0):+d} |",
                f"| Total Positions | {fresh_d.get('position_count', 0)} | {hist_d.get('position_count', 0)} | "
                f"{fresh_d.get('position_count', 0) - hist_d.get('position_count', 0):+d} |",
                f"| Game Length (mean) | {fresh_d['game_length']['mean']:.2f} | {hist_d['game_length']['mean']:.2f} | "
                f"{fresh_d['game_length']['mean'] - hist_d['game_length']['mean']:+.2f} |",
                f"| Game Length (p50 / p90) | {fresh_d['game_length']['p50']:.1f} / {fresh_d['game_length']['p90']:.1f} | "
                f"{hist_d['game_length']['p50']:.1f} / {hist_d['game_length']['p90']:.1f} | - |",
                f"| Player 0 Fraction | {fresh_d['player_distribution']['p0_fraction']:.4f} | "
                f"{hist_d['player_distribution']['p0_fraction']:.4f} | "
                f"{fresh_d['player_distribution']['p0_fraction'] - hist_d['player_distribution']['p0_fraction']:+.4f} |",
                f"| Search Policy Entropy (mean) | {fresh_d['search_policy_entropy']['mean']:.4f} | "
                f"{hist_d['search_policy_entropy']['mean']:.4f} | "
                f"{fresh_d['search_policy_entropy']['mean'] - hist_d['search_policy_entropy']['mean']:+.4f} |",
                f"| Incumbent Policy Entropy (mean) | {fresh_d['incumbent_policy_entropy']['mean']:.4f} | "
                f"{hist_d['incumbent_policy_entropy']['mean']:.4f} | "
                f"{fresh_d['incumbent_policy_entropy']['mean'] - hist_d['incumbent_policy_entropy']['mean']:+.4f} |",
                f"| Legal L1 (mean) | {fresh_d['legal_l1_search_vs_incumbent']['mean']:.4f} | "
                f"{hist_d['legal_l1_search_vs_incumbent']['mean']:.4f} | "
                f"{fresh_d['legal_l1_search_vs_incumbent']['mean'] - hist_d['legal_l1_search_vs_incumbent']['mean']:+.4f} |",
                f"| Legal L1 (p50 / p90 / p95 / p99) | "
                f"{fresh_d['legal_l1_search_vs_incumbent']['p50']:.4f} / {fresh_d['legal_l1_search_vs_incumbent']['p90']:.4f} / {fresh_d['legal_l1_search_vs_incumbent']['p95']:.4f} / {fresh_d['legal_l1_search_vs_incumbent']['p99']:.4f} | "
                f"{hist_d['legal_l1_search_vs_incumbent']['p50']:.4f} / {hist_d['legal_l1_search_vs_incumbent']['p90']:.4f} / {hist_d['legal_l1_search_vs_incumbent']['p95']:.4f} / {hist_d['legal_l1_search_vs_incumbent']['p99']:.4f} | - |",
                f"| Legal JS Divergence (mean) | {fresh_d['legal_js_search_vs_incumbent']['mean']:.4f} | "
                f"{hist_d['legal_js_search_vs_incumbent']['mean']:.4f} | "
                f"{fresh_d['legal_js_search_vs_incumbent']['mean'] - hist_d['legal_js_search_vs_incumbent']['mean']:+.4f} |",
                f"| Legal JS (p50 / p90 / p95 / p99) | "
                f"{fresh_d['legal_js_search_vs_incumbent']['p50']:.4f} / {fresh_d['legal_js_search_vs_incumbent']['p90']:.4f} / {fresh_d['legal_js_search_vs_incumbent']['p95']:.4f} / {fresh_d['legal_js_search_vs_incumbent']['p99']:.4f} | "
                f"{hist_d['legal_js_search_vs_incumbent']['p50']:.4f} / {hist_d['legal_js_search_vs_incumbent']['p90']:.4f} / {hist_d['legal_js_search_vs_incumbent']['p95']:.4f} / {hist_d['legal_js_search_vs_incumbent']['p99']:.4f} | - |",
                f"| Top-1 Disagreement Rate | {fresh_d['top1_disagreement_rate']:.4f} | "
                f"{hist_d['top1_disagreement_rate']:.4f} | "
                f"{fresh_d['top1_disagreement_rate'] - hist_d['top1_disagreement_rate']:+.4f} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Training & Target Metrics (Fresh Validation Probe)",
            "",
            "| Lane | Step | CE(search) | CE(incumbent) | CE(mixed) | Search-CE Improv | Fit Fraction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    probe = summary.get("probe_target_metrics") or {}
    for lane in lanes:
        for step in summary["checkpoint_steps"]:
            entry = probe.get(lane, {}).get(str(step))
            if entry is None:
                continue
            fit = entry["fit_fraction"]
            fit_str = "n/a" if fit is None else f"{fit:.4f}"
            lines.append(
                f"| {lane} | {step} | {entry['ce_candidate_search']:.4f} | "
                f"{entry['ce_candidate_incumbent']:.4f} | "
                f"{entry['ce_candidate_mixed']:.4f} | "
                f"{entry['search_target_ce_improvement_vs_incumbent']:+.4f} | "
                f"{fit_str} |"
            )

    lines.extend(
        [
            "",
            "## Policy Drift vs Incumbent (Fresh Validation Probe)",
            "",
            "| Lane | Step | L1 mean | L1 max | L1 p50 | L1 p90 | L1 p95 | L1 p99 | JS mean | Top-1 Change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane in lanes:
        for step in summary["checkpoint_steps"]:
            entry = probe.get(lane, {}).get(str(step))
            if entry is None:
                continue
            d = entry["drift_vs_incumbent"]
            lines.append(
                f"| {lane} | {step} | {d['legal_l1_mean']:.6f} | {d['legal_l1_max']:.6f} | "
                f"{d['legal_l1_p50']:.6f} | {d['legal_l1_p90']:.6f} | "
                f"{d['legal_l1_p95']:.6f} | {d['legal_l1_p99']:.6f} | "
                f"{d['legal_js_mean']:.6f} | {d['top1_change_rate']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Parameter Drift (Relative L2 Drift vs Incumbent)",
            "",
            "| Lane | Step | Trunk | Policy Head | Value Stack |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    drift = summary.get("drift") or {}
    for lane in lanes:
        for step in summary["checkpoint_steps"]:
            entry = drift.get(lane, {}).get(str(step))
            if entry is None:
                continue

            def _extract_drift(v: Any) -> float:
                if isinstance(v, dict):
                    return float(v.get("relative", v.get("l2_update_norm", 0.0)))
                return float(v) if v is not None else 0.0

            t_val = _extract_drift(entry.get("trunk", 0.0))
            p_val = _extract_drift(entry.get("policy_head", 0.0))
            v_val = _extract_drift(entry.get("value_head", 0.0))
            lines.append(
                f"| {lane} | {step} | {t_val:.6f} | {p_val:.6f} | {v_val:.6f} |"
            )

    lines.extend(
        [
            "",
            f"## Search Diagnostics ({PUCT_CONTEXT} context, candidate vs incumbent)",
            "",
            "| Lane | Step | Move Change Rate | Visit JS | Q-Rank Change | Root-Value Delta |",
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
                "## Per-Depth Policy L1/JS on Expanded Probe States "
                f"({PUCT_CONTEXT}, candidate vs incumbent)",
                "",
                "| Lane | Step | Depth | Expanded Nodes | L1 Mean | JS Mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for lane in lanes:
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
            "## Canonical Paired Arena (Candidate vs Frozen Incumbent)",
            "",
            "| Lane | Step | Context | Paired Effect | 95% CI | P0 Effect | P1 Effect | W/D/L |",
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
            ".venv/bin/python ml/alphazero_lite/run_fresh_selfplay_anchor_iteration.py \\",
            f"  --workdir {summary['inputs']['workdir']} \\",
            f"  --games {inputs['fresh_selfplay']['games_requested']} \\",
            f"  --seed {inputs['seed']} \\",
            "  --arena-workers 24",
            "```",
            "",
            "Full JSON evidence: "
            "`docs/data/alphazero-lite-fresh-selfplay-anchor-summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_selfplay_anchor"),
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=REPO_ROOT / "model-artifact/current",
    )
    parser.add_argument(
        "--historical-replay",
        type=Path,
        default=Path("/tmp/azlite_shared_trunk_learning/replay.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-selfplay-anchor-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/alphazero-lite-fresh-selfplay-anchor-results.md",
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--c-puct", type=float, default=DEFAULT_C_PUCT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument(
        "--reuse-fresh-selfplay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse existing fresh selfplay if present in workdir",
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
        help="Run 1200:1200 high-budget gate if beta_095 passes 384:256 safety gate",
    )
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_weights_sha = sha256_file(args.current / "weights.json")
    if current_weights_sha != CURRENT_HASH:
        raise RuntimeError(
            "current artifact weights do not match expected CURRENT_HASH"
        )

    # 1. Materialize incumbent checkpoint
    init_npz = materialize_weights_json_checkpoint(
        weights_path=args.current / "weights.json",
        out_path=args.workdir / "initialization.npz",
    )

    # 2. Generate fresh self-play exactly once
    fresh_replay_path = args.workdir / "fresh_self_play.jsonl"
    if args.reuse_fresh_selfplay and fresh_replay_path.is_file():
        print(f"[self_play] reusing existing fresh self-play: {fresh_replay_path}")
        fresh_rows = read_jsonl(fresh_replay_path)
        game_indices = set(
            int(r["game_index"]) for r in fresh_rows if "game_index" in r
        )
        selfplay_meta = {
            "games_requested": args.games,
            "games_generated": len(game_indices),
            "positions_generated": len(fresh_rows),
            "seed": args.seed,
            "simulations": args.simulations,
            "c_puct": args.c_puct,
            "player_mode": "puct",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "default",
            "value_target_mode": "default",
            "policy_target_noise_mode": "noisy",
            "tree_reuse_enabled": True,
            "replay_path": str(fresh_replay_path),
            "replay_sha256": sha256_file(fresh_replay_path),
            "checkpoint_npz_sha256": sha256_file(init_npz),
        }
    else:
        selfplay_meta = generate_fresh_self_play(
            out_path=fresh_replay_path,
            checkpoint_npz=init_npz,
            games=args.games,
            seed=args.seed,
            simulations=args.simulations,
            c_puct=args.c_puct,
            workers=args.workers,
        )
        fresh_rows = read_jsonl(fresh_replay_path)

    # 3. Create fresh manifest and batch plan
    audit_data = {
        "schema": f"{NAMESPACE}_replay_audit",
        "replay_rows": len(fresh_rows),
        "replay_sha256": sha256_file(fresh_replay_path),
        "current_weights_sha256": CURRENT_HASH,
        "policy_target_mode": "default",
        "value_target_mode": "default",
        "generator": f"self_play.py fresh {args.simulations}-simulation PUCT",
    }
    (args.workdir / "replay_audit.json").write_text(
        json.dumps(audit_data, indent=2) + "\n"
    )
    build_manifest(
        rows=fresh_rows,
        workdir=args.workdir,
        current=args.current,
        replay=fresh_replay_path,
        seed=args.seed,
        epochs=1,
        batch_size=args.batch_size,
        replay_audit=args.workdir / "replay_audit.json",
    )
    manifest_path = args.workdir / "training_manifest.json"
    manifest = verify_manifest(manifest_path)

    # 4. Dataset-shift diagnostics
    print("[diagnostics] computing dataset-shift diagnostics...", flush=True)
    incumbent_model = _new_model(device)
    load_checkpoint_into_model(incumbent_model, init_npz)
    incumbent_state = {
        k: v.detach().clone() for k, v in incumbent_model.state_dict().items()
    }

    historical_rows = (
        read_jsonl(args.historical_replay) if args.historical_replay.is_file() else []
    )
    dataset_shift = dataset_shift_comparison(
        fresh_rows, historical_rows, incumbent_state
    )

    # 5. Run matched training lanes
    steps = CHECKPOINT_STEPS
    lanes: dict[str, dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]] = {}
    initial_grads: dict[str, float] = {}
    for lane, beta in BETA_LANES:
        print(f"[train] running lane {lane} (beta={beta})...", flush=True)
        configure_determinism(device, args.seed)
        snapshots, init_grad = replay_lane_fresh(
            manifest, args.workdir / lane, device, beta, steps
        )
        lanes[lane] = snapshots
        initial_grads[lane] = init_grad

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

    beta100_drift = group_delta(lanes["beta_100"][46][0], incumbent)["policy_head"]
    beta100_incumbent_equivalent = beta100_drift < BETA100_DRIFT_TOL

    # 6. Validation probe & training metrics
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    validation_indexes = np.load(paths["validation_source_indexes"], allow_pickle=False)
    probe, probe_manifest = decoded_validation_manifest(fresh_rows, validation_indexes)
    probe_manifest["validation_source_indexes_sha256"] = sha256_file(
        paths["validation_source_indexes"]
    )
    probe_manifest["replay_sha256"] = sha256_file(fresh_replay_path)
    probe_manifest["manifest_sha256"] = stable_hash(probe_manifest)
    probe_rows = [fresh_rows[index] for index in probe_manifest["source_indexes"]]

    # Denominator reference from beta_000 step 46
    beta000_probe = probe_target_metrics_fresh(
        probe_rows, lanes["beta_000"], incumbent, 0.0, None
    )
    beta000_step46_search_ce = beta000_probe["46"]["ce_candidate_search"]

    probe_metrics: dict[str, Any] = {}
    for lane, beta in BETA_LANES:
        probe_metrics[lane] = probe_target_metrics_fresh(
            probe_rows, lanes[lane], incumbent, beta, beta000_step46_search_ce
        )

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

    # Export artifacts
    artifacts = {
        lane: export_snapshot_artifacts(snapshots, args.workdir / lane)
        for lane, snapshots in lanes.items()
    }
    incumbent_artifacts = export_snapshot_artifacts(
        {0: (incumbent, {})}, args.workdir / "incumbent"
    )

    # 7. Search diagnostics (PUCT probe)
    puct: dict[str, Any] = {}
    if args.puct:
        print("[puct] running PUCT search probe trajectories...", flush=True)
        probe_hash = stable_hash(probe_manifest)
        for lane, _beta in BETA_LANES:
            puct[lane] = puct_trajectory(
                probe[:PROBE_SIZE],
                artifacts[lane],
                args.workdir / lane,
                probe_hash,
                contexts=(PUCT_CONTEXT,),
            )

    # 8. Arena evaluation: 384:256
    arena: dict[str, Any] = {}
    if args.arena:
        print("[arena] running Stage 1 (384:256) paired arena matches...", flush=True)
        stage1_targets = [
            ("beta_000", 16, STAGE1_CONTEXT),
            ("beta_000", 46, STAGE1_CONTEXT),
            ("beta_095", 16, STAGE1_CONTEXT),
            ("beta_095", 46, STAGE1_CONTEXT),
            ("beta_100", 46, STAGE1_CONTEXT),
        ]
        arena = paired_arena(
            artifacts,
            args.current,
            args.workdir / "arena",
            args.arena_workers,
            stage1_targets,
        )

    # Stage 2: 1200:1200 high-budget gate
    if args.arena and args.stage2:
        entry_095_46 = (
            arena.get("beta_095", {}).get(str(STAGE2_STEP), {}).get(STAGE1_CONTEXT)
        )
        fit_095_46 = (
            probe_metrics.get("beta_095", {})
            .get(str(STAGE2_STEP), {})
            .get("fit_fraction")
        )
        if (
            entry_095_46 is not None
            and fit_095_46 is not None
            and arena_safe(entry_095_46)
            and fit_095_46 >= MEANINGFUL_FIT_FRACTION
        ):
            print(
                "[arena] Stage 1 passed! Running Stage 2 (1200:1200) arena for beta_095 step 46...",
                flush=True,
            )
            stage2_targets = [("beta_095", STAGE2_STEP, STAGE2_CONTEXT)]
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
        else:
            print(
                "[arena] beta_095 did not pass Stage 1 safety/fit gate for 1200:1200 arena.",
                flush=True,
            )

    # 9. Per-depth probe
    per_depth: dict[str, Any] = {}
    if args.per_depth_probe:
        print("[per_depth] running per-depth MCTS tree probes...", flush=True)
        for lane, _beta in BETA_LANES:
            per_depth[lane] = {}
            for step in (16, 46):
                cand_path = artifacts[lane][step]
                per_depth[lane][str(step)] = puct_probe(
                    probe,
                    cand_path,
                    incumbent_artifacts[0],
                    PUCT_CONTEXT,
                    modes=("incumbent_all",),
                )

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
            "current_weights_sha256": CURRENT_HASH,
            "fresh_selfplay": selfplay_meta,
            "initialization_checkpoint_sha256": sha256_file(init_npz),
            "seed": args.seed,
            "optimizer": {
                "type": "Adam",
                "lr": args.lr,
                "weight_decay": 0.0,
            },
            "gradient_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "trainable_scope": TRAINABLE_SCOPE,
            "beta_lanes": [{"lane": lane, "beta": beta} for lane, beta in BETA_LANES],
            "lane_trainable_scope": {
                lane: lane_trainable_scope(TRAINABLE_SCOPE) for lane, _ in BETA_LANES
            },
            "checkpoint_steps": steps,
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
        "sanity": {
            "lanes_start_identical": lanes_start_identical,
            "all_lanes_trunk_zero_change": all_lanes_trunk_zero,
            "all_lanes_value_stack_zero_change": all_lanes_value_stack_zero,
            "beta_100_incumbent_equivalent": beta100_incumbent_equivalent,
            "beta_100_policy_head_drift": beta100_drift,
            "beta_100_drift_tolerance": BETA100_DRIFT_TOL,
            "beta_100_initial_policy_grad_norm": initial_grads.get("beta_100", 0.0),
        },
        "dataset_shift_diagnostics": dataset_shift,
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
    summary["classification"] = classify_fresh(summary)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(summary), encoding="utf-8")
    print(f"\nClassification: {json.dumps(summary['classification'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
