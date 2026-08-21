#!/usr/bin/env python3
# ruff: noqa: E501, E701, E702
"""Test action-level robust-Q projection on the immutable PR #218 inputs."""

from __future__ import annotations

import argparse
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

from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402
from ml.alphazero_lite.fresh_p1_action_q_projection import (  # noqa: E402
    action_q_projected_teacher,
    magnitude_matched_teacher,
)
from ml.alphazero_lite.fresh_p1_adapter_teacher_audit import state_round_trips_kalah_v3  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_fresh_p1_adapter_budget_factorization import (  # noqa: E402
    A16_STATE_SHA,
    P1_CHECKPOINT_SHA,
    REPLAY_SHA,
    arena_records,
    state_hash,
    _suite,
)
from ml.alphazero_lite.run_fresh_p1_adapter_root_q_advantage import (  # noqa: E402
    PR217_BATCH_MANIFEST_SHA,
    drift,
)
from ml.alphazero_lite.run_fresh_p1_adapter_teacher_quality_audit import canonical_hash  # noqa: E402
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (  # noqa: E402
    ADAPTER_KEYS,
    BETA,
    export,
    new_model,
    output,
)
from ml.alphazero_lite.run_fresh_p1_checkpoint_selection import SEED  # noqa: E402
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (  # noqa: E402
    _cross_entropy,
    incumbent_policy_batch,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    _batch,
    _losses,
    _save_snapshot,
)
from ml.alphazero_lite.train import (  # noqa: E402
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

LANES = ("baseline_stored384", "action_q_projected", "magnitude_matched")
CHECKPOINTS = (1, 4, 16, 46)
STEPS = 46
LR = 1e-5
ROOT_Q_SHA = "57963336d66bd7fe9b2fc83252f2a1f5fd621578a5788cdb44ec0da8cb94231f"


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        key: float(np.percentile(values, percentile))
        for key, percentile in (
            ("p50", 50),
            ("p75", 75),
            ("p90", 90),
            ("p95", 95),
            ("p99", 99),
        )
    }


def q_arrays(entry: dict[str, Any], budget: int) -> tuple[np.ndarray, np.ndarray]:
    q, visits = np.zeros(6), np.zeros(6, dtype=int)
    for child in entry[f"q{budget}"]["children"]:
        action = int(child["action"])
        q[action], visits[action] = float(child["q_value"]), int(child["visits"])
    return q, visits


def build_teachers(
    rows: list[dict[str, Any]], parent: np.ndarray, cache: dict[str, dict[str, Any]]
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    projected, controls, support = [], [], []
    audit: dict[str, list[float]] = defaultdict(list)
    expected: dict[str, dict[str, list[float]]] = {
        name: {str(budget): [] for budget in (384, 1200)}
        for name in ("stored384", "action_q_projected", "magnitude_matched")
    }
    for index, row in enumerate(rows):
        key = canonical_hash(row["state"])
        legal = np.zeros(6, dtype=bool)
        legal[[int(action) for action in row["legal_moves"]]] = True
        stored = np.asarray(row["policy"], dtype=np.float64)
        q384, visits384 = q_arrays(cache[key], 384)
        q1200, visits1200 = q_arrays(cache[key], 1200)
        target, supported = action_q_projected_teacher(
            parent[index], stored, q384, q1200, visits384, visits1200, legal
        )
        control = magnitude_matched_teacher(parent[index], stored, target)
        projected.append(target)
        controls.append(control)
        support.append(supported)
        delta = stored - parent[index]
        original_l1, projected_l1 = (
            np.abs(delta).sum(),
            np.abs(target - parent[index]).sum(),
        )
        audit["legal_actions_supported_fraction"].append(float(supported[legal].mean()))
        audit["positive_delta_actions_supported_fraction"].append(
            float(supported[(delta > 0) & legal].mean())
            if np.any((delta > 0) & legal)
            else 1.0
        )
        audit["negative_delta_actions_supported_fraction"].append(
            float(supported[(delta < 0) & legal].mean())
            if np.any((delta < 0) & legal)
            else 1.0
        )
        audit["original_teacher_l1"].append(float(original_l1))
        audit["projected_teacher_l1"].append(float(projected_l1))
        audit["retained_delta_mass"].append(float(projected_l1 / 2))
        audit["discarded_delta_mass"].append(float((original_l1 - projected_l1) / 2))
        for name, policy in (
            ("stored384", stored),
            ("action_q_projected", target),
            ("magnitude_matched", control),
        ):
            for budget, q in ((384, q384), (1200, q1200)):
                expected[name][str(budget)].append(
                    float(np.dot(policy - parent[index], q))
                )
    stored = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    result = {
        "baseline_stored384": stored,
        "action_q_projected": np.asarray(projected),
        "magnitude_matched": np.asarray(controls),
    }
    ce_original = _cross_entropy(parent, stored)
    ce_projected = _cross_entropy(parent, result["action_q_projected"])
    return (
        result,
        np.asarray(support),
        {
            "per_state": {
                key: summary(np.asarray(value)) for key, value in audit.items()
            },
            "original_ce_opportunity_retained_fraction": float(
                ce_projected.mean() / ce_original.mean()
            ),
            "expected_q_change": {
                name: {
                    budget: {
                        "mean": float(np.mean(values)),
                        **summary(np.asarray(values)),
                    }
                    for budget, values in budgets.items()
                }
                for name, budgets in expected.items()
            },
        },
    )


def final_target(
    teacher: torch.Tensor, parent: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    target = (1 - BETA) * teacher + BETA * parent
    target = torch.where(mask.bool(), target, torch.zeros_like(target))
    return target / target.sum(dim=1, keepdim=True)


def gradients(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    parent_state: dict[str, torch.Tensor],
    teachers: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    paths = {key: Path(value) for key, value in manifest["artifact_paths"].items()}
    source, plan = (
        np.load(paths["train_source_indexes"], allow_pickle=False),
        np.load(paths["batch_indexes"], allow_pickle=False),
    )
    indexes = [int(source[int(index)]) for index in plan[0] if int(index) >= 0]
    vectors: dict[str, torch.Tensor] = {}
    for lane in LANES:
        model, parent = new_model(device), new_model(device)
        model.load_state_dict(parent_state)
        parent.load_state_dict(parent_state)
        apply_trainable_scope(model, "policy_adapter_only")
        for parameter in parent.parameters():
            parameter.requires_grad_(False)
        batch = _batch(
            [rows[index] for index in indexes], np.arange(len(indexes)), device
        )
        teacher = torch.as_tensor(
            teachers[lane][indexes], dtype=torch.float32, device=device
        )
        policy, value = _losses(
            model,
            {
                **batch,
                "p": final_target(
                    teacher, incumbent_policy_batch(parent, batch), batch["mask"]
                ),
            },
        )
        (policy + value).backward()
        vectors[lane] = torch.cat(
            [
                parameter.grad.flatten().cpu()
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
        )
    return {
        "norms": {lane: float(vector.norm()) for lane, vector in vectors.items()},
        "cosines": {
            f"{left}_vs_{right}": float(
                torch.nn.functional.cosine_similarity(
                    vectors[left], vectors[right], dim=0
                )
            )
            for left, right in (
                (LANES[0], LANES[1]),
                (LANES[0], LANES[2]),
                (LANES[1], LANES[2]),
            )
        },
    }


def train_lane(
    lane: str,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    parent_state: dict[str, torch.Tensor],
    teacher: np.ndarray,
    workdir: Path,
    device: torch.device,
) -> dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]:
    paths = {key: Path(value) for key, value in manifest["artifact_paths"].items()}
    source, plan = (
        np.load(paths["train_source_indexes"], allow_pickle=False),
        np.load(paths["batch_indexes"], allow_pickle=False),
    )
    model, parent = new_model(device), new_model(device)
    model.load_state_dict(parent_state)
    parent.load_state_dict(parent_state)
    apply_trainable_scope(model, "policy_adapter_only")
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LR,
        weight_decay=0.0,
    )
    snapshots = {
        0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)
    }
    for step, batch_indexes in enumerate(plan[:STEPS], 1):
        indexes = [
            int(source[int(index)]) for index in batch_indexes if int(index) >= 0
        ]
        batch = _batch(
            [rows[index] for index in indexes], np.arange(len(indexes)), device
        )
        target = final_target(
            torch.as_tensor(teacher[indexes], dtype=torch.float32, device=device),
            incumbent_policy_batch(parent, batch),
            batch["mask"],
        )
        policy, value = _losses(model, {**batch, "p": target})
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            1.0,
        )
        optimizer.step()
        if step in CHECKPOINTS:
            snapshots[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )
    return snapshots


def action_movement(
    candidate: np.ndarray,
    parent: np.ndarray,
    support: np.ndarray,
    cache: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    absolute = np.abs(candidate - parent)
    hashes = [canonical_hash(row["state"]) for row in rows]
    unvisited384 = np.asarray([q_arrays(cache[key], 384)[1] == 0 for key in hashes])
    unvisited1200 = np.asarray([q_arrays(cache[key], 1200)[1] == 0 for key in hashes])
    total = absolute.sum()
    if total == 0:
        return {
            "robust_q_supported": 0.0,
            "q_unsupported": 0.0,
            "unvisited_384": 0.0,
            "unvisited_1200": 0.0,
        }
    return {
        "robust_q_supported": float(absolute[support].sum() / total),
        "q_unsupported": float(absolute[~support].sum() / total),
        "unvisited_384": float(absolute[unvisited384].sum() / total),
        "unvisited_1200": float(absolute[unvisited1200].sum() / total),
    }


def value_support(
    candidate: np.ndarray,
    parent: np.ndarray,
    cache: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for budget in (384, 1200):
        values, positive, negative = [], [], []
        for index, row in enumerate(rows):
            key = canonical_hash(row["state"])
            delta, q = candidate[index] - parent[index], q_arrays(cache[key], budget)[0]
            contribution = delta * q
            values.append(float(contribution.sum()))
            positive.append(float(np.maximum(contribution, 0).sum()))
            negative.append(float(np.minimum(contribution, 0).sum()))
        array = np.asarray(values)
        result[str(budget)] = {
            "mean": float(array.mean()),
            "positive_fraction": float(np.mean(array > 0)),
            "p10": float(np.percentile(array, 10)),
            "p25": float(np.percentile(array, 25)),
            "p50": float(np.percentile(array, 50)),
            "p75": float(np.percentile(array, 75)),
            "p90": float(np.percentile(array, 90)),
            "positive_contribution_mass": float(np.mean(positive)),
            "negative_contribution_mass": float(np.mean(negative)),
        }
    return result


def metrics(
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    parent_state: dict[str, torch.Tensor],
    pure_state: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    teachers: dict[str, np.ndarray],
    lane: str,
    support: np.ndarray,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    parent, stored, lane_teacher = (
        output(parent_state, x, mask),
        teachers["baseline_stored384"],
        teachers[lane],
    )
    original_opportunity = float(np.mean(_cross_entropy(parent, stored)))
    own_opportunity = float(np.mean(_cross_entropy(parent, lane_teacher)))
    pure = output(pure_state, x, mask)
    original_full_fit = original_opportunity - float(
        np.mean(_cross_entropy(pure, stored))
    )
    own_full_fit = own_opportunity - float(np.mean(_cross_entropy(pure, lane_teacher)))
    result = {}
    for step, (state, _) in snapshots.items():
        candidate = output(state, x, mask)
        original_ce = float(np.mean(_cross_entropy(candidate, stored)))
        own_ce = float(np.mean(_cross_entropy(candidate, lane_teacher)))
        result[str(step)] = {
            "ce_candidate_stored384": original_ce,
            "ce_candidate_p1": float(np.mean(_cross_entropy(candidate, parent))),
            "ce_candidate_lane_target": own_ce,
            "fit_fraction_original_teacher": None
            if not step
            else (original_opportunity - original_ce) / original_full_fit,
            "fit_fraction_lane_teacher": None
            if not step
            else (own_opportunity - own_ce) / own_full_fit,
            "drift": drift(candidate, parent, mask),
            "action_learned_movement": action_movement(
                candidate, parent, support, cache, rows
            ),
            "value_support": value_support(candidate, parent, cache, rows),
        }
    return result


def arena_effect(
    challenger: Path,
    current: Path,
    context: str,
    role: str,
    workdir: Path,
    workers: int,
    suite_hash: str,
) -> dict[str, Any]:
    control = arena_records(
        workdir=workdir / "control",
        challenger=current,
        current=current,
        context=context,
        role=f"{current.name}_control",
        workers=workers,
        suite_hash=suite_hash,
    )
    records = arena_records(
        workdir=workdir,
        challenger=challenger,
        current=current,
        context=context,
        role=role,
        workers=workers,
        suite_hash=suite_hash,
    )
    result = paired_opening_candidate_effect(records, control)
    return {
        key: result[key]
        for key in (
            "paired_candidate_effect",
            "opening_bootstrap_ci",
            "p0_effect",
            "p1_effect",
        )
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# PR #218 Action-Level Robust-Q Projection",
        "",
        f"**Classification:** `{result['classification']}`",
        "",
        f"**Recommended next experiment:** {result['recommended_next_experiment']}",
        "",
        "## Pre-Training Audit",
        "",
        f"Original CE opportunity retained: {result['pre_training_audit']['original_ce_opportunity_retained_fraction']:.4f}",
        "",
        "| Metric | P50 | P75 | P90 | P95 | P99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in result["pre_training_audit"]["per_state"].items():
        lines.append(
            f"| {name} | {values['p50']:.4f} | {values['p75']:.4f} | {values['p90']:.4f} | {values['p95']:.4f} | {values['p99']:.4f} |"
        )
    lines += [
        "",
        "## Expected-Q Change",
        "",
        "| Teacher | Q budget | Mean | P50 | P90 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, budgets in result["pre_training_audit"]["expected_q_change"].items():
        for budget, values in budgets.items():
            lines.append(
                f"| {name} | {budget} | {values['mean']:+.6f} | {values['p50']:+.6f} | {values['p90']:+.6f} |"
            )
    lines += [
        "",
        "## Gradient Direction",
        "",
        "```json",
        json.dumps(result["initial_gradients"], indent=2, sort_keys=True),
        "```",
        "",
        "## Training",
        "",
        "| Lane | Step | Original fit | Lane fit | Mean L1 | Q-supported movement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in LANES:
        for step in CHECKPOINTS:
            metric = result["lanes"][lane]["metrics"][str(step)]
            lines.append(
                f"| {lane} | {step} | {metric['fit_fraction_original_teacher']:.4f} | {metric['fit_fraction_lane_teacher']:.4f} | {metric['drift']['mean_l1']:.6f} | {metric['action_learned_movement']['robust_q_supported']:.4f} |"
            )
    lines += [
        "",
        "## Post-Training Value Support",
        "",
        "| Lane | Step | Q budget | Mean | Positive fraction | Positive mass | Negative mass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in LANES:
        for step in CHECKPOINTS:
            for budget, values in result["lanes"][lane]["metrics"][str(step)][
                "value_support"
            ].items():
                lines.append(
                    f"| {lane} | {step} | {budget} | {values['mean']:+.6f} | {values['positive_fraction']:.4f} | {values['positive_contribution_mass']:.6f} | {values['negative_contribution_mass']:.6f} |"
                )
    lines += [
        "",
        "## Arena",
        "",
        "| Lane | Step | Context | Effect | 95% CI | Seat P0 | Seat P1 |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for lane, checkpoints in result["arena"].items():
        for step, contexts in checkpoints.items():
            for context, value in contexts.items():
                ci = value["opening_bootstrap_ci"]
                lines.append(
                    f"| {lane} | {step} | {context} | {value['paired_candidate_effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {value['p0_effect']:+.4f} | {value['p1_effect']:+.4f} |"
                )
    lines += [
        "",
        "## Contracts",
        "",
        "```json",
        json.dumps(
            {
                "hashes": result["hashes"],
                "invariants": result["invariants"],
                "guardrails": result["guardrails"],
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr218_action_q_projection")
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument(
        "--q-workdir", type=Path, default=Path("/tmp/azlite_pr214_root_q_advantage")
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-action-q-projection-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-action-q-projection-results.md",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    determinism = configure_determinism(device, SEED)
    p1_checkpoint = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    p1_artifact = p1_checkpoint.parent / "artifact"
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    cache_path = args.q_workdir / "root_q_cache.jsonl"
    manifest_path = args.adapter_workdir / "training_manifest.json"
    rows, cache, manifest = (
        read_jsonl(replay),
        {entry["state_hash"]: entry for entry in read_jsonl(cache_path)},
        verify_manifest(manifest_path),
    )
    model, a16 = new_model(device), new_model(device)
    load_checkpoint_into_model(model, p1_checkpoint)
    load_checkpoint_into_model(
        a16, args.adapter_workdir / "artifacts/step_0016/checkpoint.npz"
    )
    parent_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    paths = {key: Path(value) for key, value in manifest["artifact_paths"].items()}
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    source = np.load(paths["train_source_indexes"], allow_pickle=False)
    invariants = {
        "p1_checkpoint_hash": sha256_file(p1_checkpoint) == P1_CHECKPOINT_SHA,
        "replay_hash": sha256_file(replay) == REPLAY_SHA,
        "root_q_cache_hash": sha256_file(cache_path) == ROOT_Q_SHA,
        "pr217_batch_manifest_hash": sha256_file(manifest_path)
        == PR217_BATCH_MANIFEST_SHA,
        "a16_state_hash": state_hash(a16.state_dict()) == A16_STATE_SHA,
        "full_replay_state_round_trip": all(
            state_round_trips_kalah_v3(row["state"]) for row in rows
        ),
        "pr214_batch_plan": len(plan) == STEPS
        and bool(np.all((plan == -1) | ((plan >= 0) & (plan < len(source))))),
        "root_q_visit_contract": all(
            child["q_source"] == ("visited_backup" if child["visits"] else "zero_fpu")
            for entry in cache.values()
            for budget in (384, 1200)
            for child in entry[f"q{budget}"]["children"]
        ),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"immutable contract failed: {invariants}")
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    parent = output(parent_state, x, legal_mask_matrix_for_encoded_states(x))
    teachers, support, pre_training_audit = build_teachers(rows, parent, cache)
    snapshots = {
        lane: train_lane(
            lane,
            rows,
            manifest,
            parent_state,
            teachers[lane],
            args.workdir / lane,
            device,
        )
        for lane in LANES
    }
    invariants.update(
        {
            "same_initial_state": all(
                state_hash(snapshots[lane][0][0]) == state_hash(parent_state)
                for lane in LANES
            ),
            "non_adapter_parameters_bit_identical": all(
                torch.equal(state[name], parent_state[name])
                for lane in LANES
                for state, _ in snapshots[lane].values()
                for name in state
                if name not in ADAPTER_KEYS
            ),
            "projection_normalization": bool(
                np.allclose(teachers["action_q_projected"].sum(axis=1), 1.0)
            ),
            "magnitude_matched": bool(
                np.allclose(
                    np.abs(teachers["action_q_projected"] - parent).sum(axis=1),
                    np.abs(teachers["magnitude_matched"] - parent).sum(axis=1),
                )
            ),
            "filtered_actions_unchanged": bool(
                np.allclose((teachers["action_q_projected"] - parent)[~support], 0.0)
            ),
        }
    )
    pure_state = torch.load(
        args.adapter_workdir / "pure_search/snapshots/step_0046.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    lane_results = {
        lane: {
            "state_hashes": {
                str(step): state_hash(state)
                for step, (state, _) in snapshots[lane].items()
            },
            "metrics": metrics(
                snapshots[lane],
                parent_state,
                pure_state,
                rows,
                teachers,
                lane,
                support,
                cache,
            ),
        }
        for lane in LANES
    }
    suite, suite_hash = _suite()
    arena: dict[str, Any] = defaultdict(dict)
    baseline_artifact = export(
        snapshots["baseline_stored384"][46][0],
        args.workdir / "baseline_stored384/artifact_step_0046",
        "pr218_baseline_46",
    )
    for context in ("384:256", "1200:1200"):
        arena["baseline_stored384"].setdefault("46", {})[context] = arena_effect(
            baseline_artifact,
            p1_artifact,
            context,
            f"baseline_46_{context}",
            args.workdir / "arena",
            args.workers,
            suite_hash,
        )
    for lane in LANES[1:]:
        for step in (16, 46):
            artifact = export(
                snapshots[lane][step][0],
                args.workdir / lane / f"artifact_step_{step:04d}",
                f"pr218_{lane}_{step}",
            )
            arena[lane].setdefault(str(step), {})["384:256"] = arena_effect(
                artifact,
                p1_artifact,
                "384:256",
                f"{lane}_{step}",
                args.workdir / "arena",
                args.workers,
                suite_hash,
            )
            value = arena[lane][str(step)]["384:256"]
            safe = (
                value["opening_bootstrap_ci"]["lower_95"] >= -0.03
                or value["opening_bootstrap_ci"]["lower_95"]
                <= 0
                <= value["opening_bootstrap_ci"]["upper_95"]
            )
            if (
                safe
                and lane_results[lane]["metrics"][str(step)][
                    "fit_fraction_original_teacher"
                ]
                >= 0.25
            ):
                arena[lane][str(step)]["1200:1200"] = arena_effect(
                    artifact,
                    p1_artifact,
                    "1200:1200",
                    f"{lane}_{step}_1200",
                    args.workdir / "arena",
                    args.workers,
                    suite_hash,
                )
    for step, contexts in arena["action_q_projected"].items():
        p1_safe = all(
            value["opening_bootstrap_ci"]["lower_95"] >= -0.03
            or value["opening_bootstrap_ci"]["lower_95"]
            <= 0
            <= value["opening_bootstrap_ci"]["upper_95"]
            for value in contexts.values()
        )
        if {"384:256", "1200:1200"}.issubset(contexts) and p1_safe:
            artifact = export(
                snapshots["action_q_projected"][int(step)][0],
                args.workdir / "action_q_projected" / f"artifact_step_{int(step):04d}",
                f"pr218_action_q_projected_{step}",
            )
            for context in ("384:256", "1200:1200"):
                contexts[f"vs_p0_{context}"] = arena_effect(
                    artifact,
                    REPO_ROOT / "model-artifact/current",
                    context,
                    f"action_q_projected_{step}_p0",
                    args.workdir / "arena_p0",
                    args.workers,
                    suite_hash,
                )
    baseline = arena["baseline_stored384"]["46"]
    invariants["baseline_reproduced"] = (
        baseline["384:256"]["paired_candidate_effect"] == -0.01953125
        and baseline["1200:1200"]["paired_candidate_effect"] == -0.0234375
    )
    invariants["all_passed"] = all(invariants.values())
    q46, m46 = (
        lane_results["action_q_projected"]["metrics"]["46"],
        lane_results["magnitude_matched"]["metrics"]["46"],
    )
    q_arena = arena["action_q_projected"].get("46", {})
    q_safe = all(
        value["opening_bootstrap_ci"]["lower_95"] >= -0.03
        or value["opening_bootstrap_ci"]["lower_95"]
        <= 0
        <= value["opening_bootstrap_ci"]["upper_95"]
        for name, value in q_arena.items()
        if not name.startswith("vs_p0_")
    )
    q_p0 = [value for name, value in q_arena.items() if name.startswith("vs_p0_")]
    q_outperforms_control = (
        q_arena["384:256"]["paired_candidate_effect"]
        > arena["magnitude_matched"]["46"]["384:256"]["paired_candidate_effect"] + 0.01
    )
    if not invariants["all_passed"]:
        classification, next_experiment = (
            "invariant_failure",
            "Repair the immutable-input, normalization, visit, baseline, or magnitude-control contract.",
        )
    elif q46["fit_fraction_original_teacher"] < 0.25:
        classification, next_experiment = (
            "action_filter_too_sparse",
            "Assess root-Q estimate completeness before any further target engineering.",
        )
    elif (
        q_safe
        and q_outperforms_control
        and q46["action_learned_movement"]["robust_q_supported"]
        > m46["action_learned_movement"]["robust_q_supported"]
        and q_p0
        and all(
            value["opening_bootstrap_ci"]["lower_95"] >= -0.03
            or value["opening_bootstrap_ci"]["lower_95"]
            <= 0
            <= value["opening_bootstrap_ci"]["upper_95"]
            for value in q_p0
        )
    ):
        classification, next_experiment = (
            "action_q_support_rescues_update",
            "Apply the action-level Q projection prospectively on fresh self-play.",
        )
    elif all(
        abs(
            q_arena[context]["paired_candidate_effect"]
            - arena["magnitude_matched"]["46"][context]["paired_candidate_effect"]
        )
        <= 0.01
        for context in ("384:256", "1200:1200")
    ):
        classification, next_experiment = (
            "supervision_magnitude_explains_rescue",
            "Do not build an expensive Q-gated target pipeline.",
        )
    elif q46["action_learned_movement"]["q_unsupported"] > 0.5 and not q_safe:
        classification, next_experiment = (
            "parameter_sharing_reintroduces_unsupported_actions",
            "Move to an action-local residual parameterization or explicit output constraint.",
        )
    elif (
        q46["action_learned_movement"]["robust_q_supported"]
        > m46["action_learned_movement"]["robust_q_supported"]
        and not q_safe
    ):
        classification, next_experiment = (
            "action_q_sign_not_predictive",
            "Run first-divergence MCTS trajectory attribution on the failing arena openings and states.",
        )
    else:
        classification, next_experiment = (
            "inconclusive",
            "Inspect the preregistered action and arena diagnostics without adding target variants.",
        )
    result = {
        "schema": "azlite_pr218_action_q_projection_v1",
        "classification": classification,
        "recommended_next_experiment": next_experiment,
        "guardrails": {
            "fresh_self_play_generated": False,
            "promotion": False,
            "beta": BETA,
            "optimizer": "Adam",
            "lr": LR,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "batch_size": 512,
            "steps": STEPS,
            "trainable_parameters": list(ADAPTER_KEYS),
        },
        "hashes": {
            "p1_checkpoint": sha256_file(p1_checkpoint),
            "replay": sha256_file(replay),
            "root_q_cache": sha256_file(cache_path),
            "batch_manifest": sha256_file(manifest_path),
            "arena_suite": suite_hash,
        },
        "invariants": invariants,
        "determinism": determinism,
        "pre_training_audit": pre_training_audit,
        "initial_gradients": gradients(rows, manifest, parent_state, teachers, device),
        "lanes": lane_results,
        "arena": dict(arena),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_report.write_text(render(result), encoding="utf-8")
    print(classification)


if __name__ == "__main__":
    main()
