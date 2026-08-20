#!/usr/bin/env python3
# ruff: noqa: E402
"""Retrospective selective-target Gen-2 policy update experiment.

The P2-derived risk masks are frozen before retraining and are never informed by
arena results.  This script consumes the exact PR #204 replay and batch plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (  # noqa: E402
    CHECKPOINT_STEPS,
    PROBE_SIZE,
    PUCT_CONTEXT,
    TRAINABLE_SCOPE,
    _batch,
    _cross_entropy,
    _losses,
    _save_snapshot,
    _win_draw_loss,
    arena_safe,
    assert_legal_distribution,
    export_snapshot_artifacts,
    group_parameters_identical,
    incumbent_policy_batch,
    tensors_identical,
    trunk_parameters_identical,
    puct_probe,
    puct_trajectory,
)
from ml.alphazero_lite.run_frozen_trunk_head_isolation_ablation import (
    VALUE_STACK_PREFIXES,
)  # noqa: E402
from ml.alphazero_lite.run_gen2_hard_trust_region import (  # noqa: E402
    PR204_GEN2_REPLAY_HASH,
    PR204_UNPROJECTED_S46_STATE_HASH,
)
from ml.alphazero_lite.run_gen2_selfplay_anchor_iteration import (  # noqa: E402
    P1_EXPECTED_STATE_HASH,
    reconstruct_and_freeze_p1,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _new_model  # noqa: E402
from ml.alphazero_lite.run_policy_detached_trunk_ablation import _arena_records  # noqa: E402
from ml.alphazero_lite.run_gen2_tail_prior_override import probe_lane  # noqa: E402
from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import (  # noqa: E402
    decode_state,
)
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    model_outputs,
    stable_hash,
)
from ml.alphazero_lite.train import (  # noqa: E402
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

NAMESPACE = "azlite_gen2_selective_risk_training_v1"
LANES = ("beta095", "risk_q90", "risk_q75", "matched_random25")
CONTROL_SEED = 20825
Q75 = 0.008493570293918201
Q90 = 0.011032855853303441


def state_hash(row: dict[str, Any]) -> str:
    """Stable identity for an encoded board; duplicate rows deliberately collide."""
    return hashlib.sha256(
        np.asarray(row["state"], dtype=np.float32).tobytes()
    ).hexdigest()


def legal_normalize(policy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.where(mask.astype(bool), policy, 0.0)
    return out / out.sum(axis=1, keepdims=True)


def legal_l1(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left - right).sum(axis=1)


def build_risk_masks(
    rows: list[dict[str, Any]], p1: dict[str, torch.Tensor], p2: dict[str, torch.Tensor]
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    """Freeze PR #208 state scores and masks once, on unique replay boards."""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(state_hash(row), row)
    ordered_hashes = sorted(unique)
    x = np.asarray([unique[key]["state"] for key in ordered_hashes], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    p1_policy, _ = model_outputs(p1, x, mask)
    p2_policy, _ = model_outputs(p2, x, mask)
    scores = legal_l1(p2_policy, p1_policy).astype(np.float64)
    q75, q90 = float(np.percentile(scores, 75)), float(np.percentile(scores, 90))
    # Lock the published thresholds as a reproduction contract, not candidates' quantiles.
    if not (np.isclose(q75, Q75, atol=1e-10) and np.isclose(q90, Q90, atol=1e-10)):
        raise RuntimeError(f"PR #208 quantiles differ: q75={q75}, q90={q90}")
    risk75 = {key for key, score in zip(ordered_hashes, scores) if score >= q75}
    risk90 = {key for key, score in zip(ordered_hashes, scores) if score >= q90}
    parent_search = legal_l1(
        p1_policy, np.asarray([unique[k]["policy"] for k in ordered_hashes])
    )
    strata: dict[tuple[int, int, int, int], list[str]] = defaultdict(list)
    ply_by_hash: dict[str, int] = {}
    game_ply: dict[int, int] = defaultdict(int)
    for row in rows:
        key = state_hash(row)
        ply_by_hash.setdefault(key, game_ply[int(row["game_index"])])
        game_ply[int(row["game_index"])] += 1
    edges = np.quantile(parent_search, [0.25, 0.5, 0.75])
    for index, key in enumerate(ordered_hashes):
        row = unique[key]
        ply = ply_by_hash[key]
        strata[
            (
                int(row.get("player", 0)),
                int(mask[index].sum()),
                ply // 10,
                int(np.searchsorted(edges, parent_search[index])),
            )
        ].append(key)
    rng = np.random.default_rng(CONTROL_SEED)
    control: set[str] = set()
    target = len(risk75)
    total = len(ordered_hashes)
    for keys in strata.values():
        count = int(round(target * len(keys) / total))
        control.update(
            rng.choice(keys, size=min(count, len(keys)), replace=False).tolist()
        )
    remaining = [key for key in ordered_hashes if key not in control]
    if len(control) < target:
        control.update(
            rng.choice(remaining, size=target - len(control), replace=False).tolist()
        )
    elif len(control) > target:
        control = set(rng.choice(sorted(control), size=target, replace=False).tolist())
    payload = {
        "state_hashes": ordered_hashes,
        "state_hashes_sha256": hashlib.sha256(
            "".join(ordered_hashes).encode()
        ).hexdigest(),
        "score_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
        "scores": scores.tolist(),
        "thresholds": {"q75": q75, "q90": q90},
        "counts": {
            "unique": len(ordered_hashes),
            "risk_q75": len(risk75),
            "risk_q90": len(risk90),
        },
        "mask_sha256": {
            name: hashlib.sha256("".join(sorted(values)).encode()).hexdigest()
            for name, values in {
                "risk_q75": risk75,
                "risk_q90": risk90,
                "matched_random25": control,
            }.items()
        },
        "matched_control_seed": CONTROL_SEED,
    }
    return payload, {
        "beta095": set(),
        "risk_q90": risk90,
        "risk_q75": risk75,
        "matched_random25": control,
    }


def statewise_targets(
    search: torch.Tensor,
    parent: torch.Tensor,
    mask: torch.Tensor,
    protected: np.ndarray,
) -> torch.Tensor:
    """Use P1 exactly on protected rows, beta=.95 mixture elsewhere."""
    protected = torch.as_tensor(protected, dtype=torch.bool, device=search.device)
    mixed = 0.05 * search + 0.95 * parent
    mixed = torch.where(protected[:, None], parent, mixed)
    mixed = torch.where(mask.bool(), mixed, torch.zeros_like(mixed))
    return mixed / mixed.sum(dim=1, keepdim=True)


def train_lane(
    manifest: dict[str, Any],
    workdir: Path,
    device: torch.device,
    p1_path: Path,
    protected: set[str],
) -> dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]:
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source, plan = (
        np.load(paths["train_source_indexes"]),
        np.load(paths["batch_indexes"]),
    )
    model, parent = _new_model(device), _new_model(device)
    load_checkpoint_into_model(model, p1_path)
    load_checkpoint_into_model(parent, p1_path)
    apply_trainable_scope(model, TRAINABLE_SCOPE)
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0)
    saved = {0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)}
    model.train()
    for step, indexes in enumerate(plan, 1):
        batch_rows = [rows[int(source[i])] for i in indexes if i >= 0]
        batch = _batch(batch_rows, np.arange(len(batch_rows)), device)
        parent_policy = incumbent_policy_batch(parent, batch)
        flags = torch.tensor(
            [state_hash(row) in protected for row in batch_rows], device=device
        )
        batch = {
            **batch,
            "p": statewise_targets(batch["p"], parent_policy, batch["mask"], flags),
        }
        assert_legal_distribution(
            batch["p"].detach().cpu().numpy(), batch["mask"].cpu().numpy()
        )
        policy, value = _losses(model, batch)
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            saved[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )
    return saved


def refreshed_top_fraction_mask(
    rows: list[dict[str, Any]],
    candidate: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
    count: int,
) -> tuple[set[str], dict[str, Any]]:
    """Deterministically select the current highest-drift unique replay states."""
    unique = {state_hash(row): row for row in rows}
    keys = sorted(unique)
    x = np.asarray([unique[key]["state"] for key in keys], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    candidate_policy, _ = model_outputs(candidate, x, mask)
    parent_policy, _ = model_outputs(parent, x, mask)
    scores = legal_l1(candidate_policy, parent_policy).astype(np.float64)
    # Stable state-hash tie breaking prevents platform-dependent mask membership.
    order = np.lexsort((np.asarray(keys), -scores))
    selected = {keys[index] for index in order[:count]}
    return selected, {
        "state_hashes_sha256": hashlib.sha256("".join(keys).encode()).hexdigest(),
        "score_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
        "mask_sha256": hashlib.sha256("".join(sorted(selected)).encode()).hexdigest(),
        "protected_unique_states": len(selected),
        "selection_threshold": float(scores[order[count - 1]]),
    }


def train_dynamic_refresh_lane(
    manifest: dict[str, Any],
    workdir: Path,
    device: torch.device,
    p1_path: Path,
    initial_protected: set[str],
    refresh_after_steps: tuple[int, ...],
) -> tuple[dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]], dict[str, Any]]:
    """Refresh a fixed-size output-risk mask after predeclared optimizer steps."""
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(paths["train_source_indexes"])
    plan = np.load(paths["batch_indexes"])
    model, parent = _new_model(device), _new_model(device)
    load_checkpoint_into_model(model, p1_path)
    load_checkpoint_into_model(parent, p1_path)
    apply_trainable_scope(model, TRAINABLE_SCOPE)
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0)
    saved = {0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)}
    protected = set(initial_protected)
    refreshes: dict[str, Any] = {
        "initial": {
            "mask_sha256": hashlib.sha256(
                "".join(sorted(protected)).encode()
            ).hexdigest(),
            "protected_unique_states": len(protected),
            "source": "frozen_pr208_q75",
        }
    }
    model.train()
    for step, indexes in enumerate(plan, 1):
        batch_rows = [rows[int(source[i])] for i in indexes if i >= 0]
        batch = _batch(batch_rows, np.arange(len(batch_rows)), device)
        parent_policy = incumbent_policy_batch(parent, batch)
        flags = np.asarray([state_hash(row) in protected for row in batch_rows])
        batch = {
            **batch,
            "p": statewise_targets(batch["p"], parent_policy, batch["mask"], flags),
        }
        policy, value = _losses(model, batch)
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            saved[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )
        if step in refresh_after_steps:
            protected, telemetry = refreshed_top_fraction_mask(
                rows, model.state_dict(), parent.state_dict(), len(initial_protected)
            )
            refreshes[str(step)] = telemetry
    return saved, refreshes


def distribution_metrics(
    candidate: np.ndarray, p1: np.ndarray, mask: np.ndarray, selected: np.ndarray
) -> dict[str, Any]:
    l1 = legal_l1(candidate, p1)
    m = selected

    def block(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_l1": float(values.mean()),
            "p50_l1": float(np.percentile(values, 50)),
            "p75_l1": float(np.percentile(values, 75)),
            "p90_l1": float(np.percentile(values, 90)),
            "p95_l1": float(np.percentile(values, 95)),
            "p99_l1": float(np.percentile(values, 99)),
            "max_l1": float(values.max()),
        }

    out = block(l1)
    out["protected"] = block(l1[m]) if m.any() else None
    out["unprotected"] = block(l1[~m]) if (~m).any() else None
    midpoint = np.maximum((candidate + p1) / 2, 1e-12)
    out["mean_js"] = float(
        np.mean(
            0.5
            * (
                np.sum(
                    candidate * np.log(np.clip(candidate / midpoint, 1e-12, None)),
                    axis=1,
                )
                + np.sum(p1 * np.log(np.clip(p1 / midpoint, 1e-12, None)), axis=1)
            )
        )
    )
    out["top1_disagreement"] = float(
        np.mean(
            np.argmax(np.where(mask, candidate, -np.inf), 1)
            != np.argmax(np.where(mask, p1, -np.inf), 1)
        )
    )
    return out


def _stratified_drift(
    rows: list[dict[str, Any]],
    keys: list[str],
    l1: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Summarize frozen-state drift by replay-only covariates."""
    unique = {state_hash(row): row for row in rows}
    ply_by_hash: dict[str, int] = {}
    game_ply: dict[int, int] = defaultdict(int)
    for row in rows:
        key = state_hash(row)
        ply_by_hash.setdefault(key, game_ply[int(row["game_index"])])
        game_ply[int(row["game_index"])] += 1
    x = np.asarray([unique[key]["state"] for key in keys], dtype=np.float32)
    legal_counts = legal_mask_matrix_for_encoded_states(x).sum(axis=1)
    groups: dict[str, np.ndarray] = {
        "player_0": np.asarray([int(unique[k].get("player", 0)) == 0 for k in keys]),
        "player_1": np.asarray([int(unique[k].get("player", 0)) == 1 for k in keys]),
    }
    for count in sorted(set(int(v) for v in legal_counts)):
        groups[f"legal_moves_{count}"] = legal_counts == count
    for bucket in sorted(set(ply_by_hash[k] // 10 for k in keys)):
        groups[f"ply_{bucket * 10}_{bucket * 10 + 9}"] = np.asarray(
            [ply_by_hash[k] // 10 == bucket for k in keys]
        )
    return {
        name: {"count": int(select.sum()), "mean_l1": float(l1[select].mean())}
        for name, select in groups.items()
        if select.any()
    }


def metrics(
    rows: list[dict[str, Any]],
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    p1: dict[str, torch.Tensor],
    p2: dict[str, torch.Tensor],
    masks: dict[str, set[str]],
    lane: str,
    beta095_ce: float,
) -> dict[str, Any]:
    unique = {state_hash(row): row for row in rows}
    keys = sorted(unique)
    x = np.asarray([unique[k]["state"] for k in keys], np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    search = np.asarray([unique[k]["policy"] for k in keys], np.float64)
    parent, _ = model_outputs(p1, x, mask)
    p2_policy, _ = model_outputs(p2, x, mask)
    protected = np.asarray([key in masks[lane] for key in keys])
    target = np.where(protected[:, None], parent, 0.05 * search + 0.95 * parent)
    target = legal_normalize(target, mask)
    p1_ce = float(np.mean(_cross_entropy(parent, search)))
    result: dict[str, Any] = {}
    for step, (state, _) in snapshots.items():
        candidate, _ = model_outputs(state, x, mask)
        ce_search = float(np.mean(_cross_entropy(candidate, search)))
        improve = p1_ce - ce_search
        l1 = legal_l1(candidate, parent)
        top25 = set(np.asarray(keys)[np.argsort(l1)[-len(masks["risk_q75"]) :]])
        top90 = set(np.asarray(keys)[np.argsort(l1)[-len(masks["risk_q90"]) :]])
        result[str(step)] = {
            "ce_candidate_search": ce_search,
            "ce_candidate_p1": float(np.mean(_cross_entropy(candidate, parent))),
            "ce_candidate_target": float(np.mean(_cross_entropy(candidate, target))),
            "search_target_ce_improvement_vs_p1": improve,
            "fit_fraction": improve / (p1_ce - beta095_ce)
            if step and p1_ce > beta095_ce
            else None,
            "search_improvement_protected": float(
                np.mean(_cross_entropy(parent[protected], search[protected]))
                - np.mean(_cross_entropy(candidate[protected], search[protected]))
            )
            if protected.any()
            else 0.0,
            "search_improvement_unprotected": float(
                np.mean(_cross_entropy(parent[~protected], search[~protected]))
                - np.mean(_cross_entropy(candidate[~protected], search[~protected]))
            )
            if (~protected).any()
            else 0.0,
            "drift": distribution_metrics(candidate, parent, mask, protected),
            "drift_by_replay_covariate": _stratified_drift(rows, keys, l1),
            "protected_leakage": {
                "initial_to_current": distribution_metrics(
                    candidate, parent, mask, protected
                )["protected"],
                "ce_candidate_p1": float(
                    np.mean(_cross_entropy(candidate[protected], parent[protected]))
                )
                if protected.any()
                else 0.0,
                # Positive alignment means indirect movement is toward P2, not P1.
                "direction_cosine_vs_p2_minus_p1": float(
                    np.dot(
                        (candidate[protected] - parent[protected]).ravel(),
                        (p2_policy[protected] - parent[protected]).ravel(),
                    )
                    / max(
                        np.linalg.norm(
                            (candidate[protected] - parent[protected]).ravel()
                        )
                        * np.linalg.norm(
                            (p2_policy[protected] - parent[protected]).ravel()
                        ),
                        1e-12,
                    )
                )
                if protected.any()
                else 0.0,
            },
            "risk_migration": {
                "jaccard_current_top25_frozen_q75": len(top25 & masks["risk_q75"])
                / len(top25 | masks["risk_q75"]),
                "fraction_current_top25_originally_protected": len(top25 & masks[lane])
                / len(top25),
                "fraction_current_top25_originally_q75_protected": len(
                    top25 & masks["risk_q75"]
                )
                / len(top25),
                "fraction_current_top25_migrated_from_q75_unprotected": len(
                    top25 - masks["risk_q75"]
                )
                / len(top25),
                "current_q75": float(np.percentile(l1, 75)),
                "current_q90": float(np.percentile(l1, 90)),
                "originally_protected_above_frozen_q75": int(
                    (l1[protected] >= Q75).sum()
                ),
                "originally_unprotected_newly_above_frozen_q75": int(
                    (l1[~protected] >= Q75).sum()
                ),
                "q90": {
                    "jaccard_current_top10_frozen_q90": len(top90 & masks["risk_q90"])
                    / len(top90 | masks["risk_q90"]),
                    "originally_protected_above_frozen_q90": int(
                        (l1[protected] >= Q90).sum()
                    ),
                    "originally_unprotected_newly_above_frozen_q90": int(
                        (
                            l1[~np.asarray([key in masks["risk_q90"] for key in keys])]
                            >= Q90
                        ).sum()
                    ),
                },
            },
        }
    return result


def classify(summary: dict[str, Any]) -> dict[str, str]:
    if not all(summary["sanity"].values()):
        return {
            "label": "invariant_failure",
            "next_experiment": "repair the failed locked-input invariant",
        }
    arena, fit = summary["arena"], summary["metrics"]
    if not arena.get("risk_q75", {}).get("46"):
        return {
            "label": "inconclusive",
            "next_experiment": "run the preregistered canonical arena matrix",
        }
    q75_safe = arena["risk_q75"]["46"]["384:256"]["safe"]
    q75_fit = fit["risk_q75"]["46"]["fit_fraction"] >= 0.25
    random_safe = arena["matched_random25"]["46"]["384:256"]["safe"]
    migration = fit["risk_q75"]["46"]["risk_migration"]
    if q75_safe and q75_fit and not random_safe:
        return {
            "label": "selective_risk_training_rescues_gen2",
            "next_experiment": "replace the retrospective P2-derived mask with a two-pass online mask without arena data",
        }
    if q75_safe and random_safe:
        return {
            "label": "generic_supervision_reduction_explains_safety",
            "next_experiment": "do not build a complicated risk detector",
        }
    if q75_safe and not q75_fit:
        return {
            "label": "selective_constraint_overregularizes",
            "next_experiment": "evaluate a less coupled parameterization before another mask",
        }
    if migration["fraction_current_top25_originally_protected"] < 0.9 and not q75_safe:
        return {
            "label": "static_mask_risk_migrates",
            "next_experiment": "test dynamic periodic mask refresh during training",
        }
    return {
        "label": "risk_mask_not_training_causal",
        "next_experiment": "inspect static-mask migration before further training interventions",
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Gen-2 Selective Risk-Target Results",
        "",
        f"**Primary classification:** `{summary['classification']['label']}`",
        "",
        f"**Recommended follow-up:** {summary['classification']['next_experiment']}",
        "",
        "## Frozen Masks",
        "",
        f"- q75: `{summary['frozen_masks']['thresholds']['q75']:.10f}`; q90: `{summary['frozen_masks']['thresholds']['q90']:.10f}`",
        f"- Unique states: `{summary['frozen_masks']['counts']['unique']}`; q75 protected: `{summary['frozen_masks']['counts']['risk_q75']}`",
        "",
        "## Training Metrics",
        "",
        "| Lane | Step | CE(search) | CE(P1) | CE(target) | Fit fraction | Unprotected improvement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in LANES:
        for step in CHECKPOINT_STEPS:
            m = summary["metrics"][lane][str(step)]
            lines.append(
                f"| {lane} | {step} | {m['ce_candidate_search']:.6f} | {m['ce_candidate_p1']:.6f} | {m['ce_candidate_target']:.6f} | {m['fit_fraction']:.4f} | {m['search_improvement_unprotected']:.6f} |"
            )
    lines += [
        "",
        "## Protected-State Leakage (Step 46)",
        "",
        "| Lane | Protected L1 mean | Protected L1 p95 | Protected L1 p99 | CE(candidate, P1) | Direction vs P2-P1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in LANES:
        leakage = summary["metrics"][lane]["46"]["protected_leakage"]
        protected = leakage["initial_to_current"] or {}
        lines.append(
            f"| {lane} | {protected.get('mean_l1', 0.0):.6f} | "
            f"{protected.get('p95_l1', 0.0):.6f} | {protected.get('p99_l1', 0.0):.6f} | "
            f"{leakage['ce_candidate_p1']:.6f} | "
            f"{leakage['direction_cosine_vs_p2_minus_p1']:+.4f} |"
        )
    lines += [
        "",
        "## Risk Migration (Step 46)",
        "",
        "| Lane | Current q75 | Top-25% Jaccard vs frozen q75 | Originally protected in current top-25% | New unprotected above frozen q75 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for lane in LANES:
        m = summary["metrics"][lane]["46"]["risk_migration"]
        lines.append(
            f"| {lane} | {m['current_q75']:.6f} | {m['jaccard_current_top25_frozen_q75']:.4f} | {m['fraction_current_top25_originally_protected']:.4f} | {m['originally_unprotected_newly_above_frozen_q75']} |"
        )
    lines += [
        "",
        "## Deterministic Search Probes vs P1",
        "",
        "| Lane | Step | Move changes | Visit JS | Q-rank changes | Root-value delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in LANES:
        for step in (16, 46):
            probe = summary["puct_vs_p1"][lane]["metrics"][str(step)][PUCT_CONTEXT]
            lines.append(
                f"| {lane} | {step} | {probe['selected_move_change_rate']:.4f} | "
                f"{probe['visit_js']:.6f} | {probe['child_q_rank_change']:+.4f} | "
                f"{probe['root_value_delta']:+.6f} |"
            )
    lines += [
        "",
        "## Frozen q75 Tail Override on risk_q75",
        "",
        "| Step | Expanded nodes | Replacement nodes | Replacement fraction |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for step, telemetry in summary["risk_q75_frozen_tail_override_probe"].items():
        lines.append(
            f"| {step} | {telemetry['total_expanded_nodes']} | "
            f"{telemetry['overridden_node_count']} | {telemetry['override_fraction']:.4f} |"
        )
    lines += [
        "",
        "## Arena",
        "",
        "| Lane | Step | Effect vs P1 | 95% CI | Safe |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for lane, entries in summary["arena"].items():
        for step, contexts in entries.items():
            e = contexts["384:256"]
            ci = e["opening_bootstrap_ci"]
            lines.append(
                f"| {lane} | {step} | {e['paired_candidate_effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {e['safe']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_gen2_selective_risk")
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--gen1-replay",
        type=Path,
        default=Path("/tmp/azlite_fresh_selfplay_anchor/fresh_self_play.jsonl"),
    )
    parser.add_argument(
        "--gen2-replay",
        type=Path,
        default=Path("/tmp/azlite_gen2_selfplay_anchor/gen2_self_play.jsonl"),
    )
    parser.add_argument(
        "--p2",
        type=Path,
        default=Path(
            "/tmp/azlite_gen2_selfplay_anchor/beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
        ),
    )
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-gen2-selective-risk-training-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-gen2-selective-risk-training-results.md",
    )
    args = parser.parse_args()
    if args.render_only:
        summary = json.loads(args.out_summary.read_text(encoding="utf-8"))
        args.out_report.write_text(render(summary), encoding="utf-8")
        return
    args.workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    configure_determinism(device, 43)
    p1_artifact, p1_path, _, _, p1_hash = reconstruct_and_freeze_p1(
        REPO_ROOT / "model-artifact/current", args.p1_workdir, args.gen1_replay, 24
    )
    p1, p2 = _new_model(device), _new_model(device)
    load_checkpoint_into_model(p1, p1_path)
    load_checkpoint_into_model(p2, args.p2)
    p1_state, p2_state = p1.state_dict(), p2.state_dict()
    rows = read_jsonl(args.gen2_replay)
    manifest = verify_manifest(
        Path("/tmp/azlite_gen2_selfplay_anchor/training_manifest.json")
    )
    frozen, masks = build_risk_masks(rows, p1_state, p2_state)
    snapshots: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for lane in LANES:
        configure_determinism(device, 43)
        snapshots[lane] = train_lane(
            manifest, args.workdir / lane, device, p1_path, masks[lane]
        )
        artifacts[lane] = export_snapshot_artifacts(
            snapshots[lane], args.workdir / lane
        )
    p2_hash = stable_hash(
        {k: v.detach().cpu().numpy().tobytes().hex() for k, v in p2_state.items()}
    )
    sanity = {
        "p1_hash": p1_hash == P1_EXPECTED_STATE_HASH,
        "p2_hash": p2_hash == PR204_UNPROJECTED_S46_STATE_HASH,
        "replay_hash": sha256_file(args.gen2_replay) == PR204_GEN2_REPLAY_HASH,
        "lanes_start_identical": all(
            tensors_identical(snapshots[lane][0][0], snapshots["beta095"][0][0])
            for lane in LANES
        ),
        "baseline_reproduced": stable_hash(
            {
                k: v.detach().cpu().numpy().tobytes().hex()
                for k, v in snapshots["beta095"][46][0].items()
            }
        )
        == PR204_UNPROJECTED_S46_STATE_HASH,
        "trunk_zero": all(
            trunk_parameters_identical(snapshots[lane][s][0], p1_state)
            for lane in LANES
            for s in CHECKPOINT_STEPS
        ),
        "value_zero": all(
            group_parameters_identical(
                snapshots[lane][s][0], p1_state, prefixes=VALUE_STACK_PREFIXES
            )
            for lane in LANES
            for s in CHECKPOINT_STEPS
        ),
        "p1_p2_frozen_stacks_identical": all(
            torch.equal(v, p2_state[k])
            for k, v in p1_state.items()
            if k.startswith(
                (
                    "input_layer.",
                    "residual_layers.",
                    "value_hidden_layer.",
                    "value_head.",
                )
            )
        ),
    }
    base_metrics = metrics(
        rows, snapshots["beta095"], p1_state, p2_state, masks, "beta095", 0.0
    )
    denominator_ce = base_metrics["46"]["ce_candidate_search"]
    all_metrics = {
        lane: metrics(
            rows, snapshots[lane], p1_state, p2_state, masks, lane, denominator_ce
        )
        for lane in LANES
    }
    print("[puct] running deterministic candidate-vs-P1 probes...", flush=True)
    probe_hash = hashlib.sha256("".join(frozen["state_hashes"]).encode()).hexdigest()
    search_probe = [
        {
            **row,
            "state": decode_state(row["state"]),
            "state_hash": state_hash(row),
            "manifest_index": index,
        }
        for index, row in enumerate(rows[:PROBE_SIZE])
    ]
    puct = {
        lane: puct_trajectory(
            search_probe,
            artifacts[lane],
            args.workdir / lane,
            probe_hash,
            contexts=(PUCT_CONTEXT,),
        )
        for lane in LANES
    }
    per_depth_puct = {
        lane: {
            str(step): puct_probe(
                search_probe,
                artifacts[lane][step],
                p1_artifact,
                PUCT_CONTEXT,
                modes=("incumbent_all",),
            )
            for step in (16, 46)
        }
        for lane in LANES
    }
    print("[tail-probe] checking frozen q75 substitution on risk_q75...", flush=True)
    tail_override = {
        str(step): probe_lane(
            rows,
            __import__(
                "ml.alphazero_lite.arena", fromlist=["ArtifactEvaluator"]
            ).ArtifactEvaluator(artifacts["risk_q75"][step]),
            __import__(
                "ml.alphazero_lite.arena", fromlist=["ArtifactEvaluator"]
            ).ArtifactEvaluator(p1_artifact),
            Q75,
        )
        for step in (16, 46)
    }
    arena: dict[str, Any] = {lane: {} for lane in LANES}
    if args.arena:
        control = _arena_records(
            args.workdir,
            p1_artifact,
            p1_artifact,
            "384:256",
            "p1_control",
            args.arena_workers,
        )
        for lane, steps in {
            "beta095": (46,),
            "risk_q90": (16, 46),
            "risk_q75": (16, 46),
            "matched_random25": (46,),
        }.items():
            for step in steps:
                effect = __import__(
                    "ml.alphazero_lite.evaluation_metrics",
                    fromlist=["paired_opening_candidate_effect"],
                ).paired_opening_candidate_effect(
                    _arena_records(
                        args.workdir / lane,
                        artifacts[lane][step],
                        p1_artifact,
                        "384:256",
                        f"{lane}_{step}",
                        args.arena_workers,
                    ),
                    control,
                )
                effect["safe"] = arena_safe(effect)
                effect["win_draw_loss"] = _win_draw_loss(
                    _arena_records(
                        args.workdir / lane,
                        artifacts[lane][step],
                        p1_artifact,
                        "384:256",
                        f"{lane}_{step}",
                        args.arena_workers,
                    )
                )
                arena[lane][str(step)] = {"384:256": effect}
    summary = {
        "schema": NAMESPACE,
        "guardrails": {
            "fresh_self_play": False,
            "arena_mask_input": False,
            "optimizer_change": False,
            "value_or_trunk_training": False,
        },
        "inputs": {
            "p0_weights_sha256": sha256_file(
                REPO_ROOT / "model-artifact/current/weights.json"
            ),
            "p1_state_hash": p1_hash,
            "p2_state_hash": p2_hash,
            "gen2_replay_sha256": sha256_file(args.gen2_replay),
            "seed": 43,
            "optimizer": {"type": "Adam", "lr": 1e-5, "weight_decay": 0.0},
            "gradient_clip": 1.0,
            "batch_size": 512,
            "checkpoint_steps": CHECKPOINT_STEPS,
        },
        "frozen_masks": frozen,
        "sanity": sanity,
        "metrics": all_metrics,
        "puct_vs_p1": puct,
        "per_depth_puct_vs_p1": per_depth_puct,
        "risk_q75_frozen_tail_override_probe": tail_override,
        "arena": arena,
    }
    summary["classification"] = classify(summary)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    args.out_report.write_text(render(summary))
    print(json.dumps(summary["classification"], indent=2))


if __name__ == "__main__":
    main()
