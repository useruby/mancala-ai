#!/usr/bin/env python3
# ruff: noqa: E402, E501, E701, E702
"""Test whether root-Q-supported stored targets avoid PR #214 adapter harm."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.fresh_p1_adapter_teacher_audit import (  # noqa: E402
    decode_kalah_v3_base_state,
    state_round_trips_kalah_v3,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
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
    state_hash as model_state_hash,
    _suite,
)
from ml.alphazero_lite.run_fresh_p1_adapter_teacher_quality_audit import (  # noqa: E402
    canonical_hash,
    entropy,
    move_index_bucket,
    state_seed,
)
from ml.alphazero_lite.run_fresh_p1_adapter_teacher_target_retrain import (  # noqa: E402
    build_clean_target_cache,
)
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
from ml.alphazero_lite.run_search_q_value_attribution_audit import (  # noqa: E402
    verify_puct_q_semantics,
)
from ml.alphazero_lite.self_play import PUCT  # noqa: E402
from ml.alphazero_lite.train import (  # noqa: E402
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402

LANES = ("baseline_stored384", "robust_advantage", "matched_random")
CHECKPOINTS = (1, 4, 16, 46)
STEPS = 46
LR = 1e-5
CONTROL_SEED = 217
GROUPS = ("robust_positive", "budget_conflicted", "robust_nonpositive")
PR217_BATCH_MANIFEST_SHA = (
    "4f476379b1ae4a5687591a31d6726f943643d6e3b2c36d72fa233d37613892c3"
)
PR217_CLEAN_TARGET_CACHE_SHA = (
    "2e8f9168db67297505b6ad8b0059d4aa521214d32adcc50191f40c7f8112fd16"
)


def unique_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Return canonical encoded-state identities without discarding row multiplicity."""
    by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_hash.setdefault(canonical_hash(row["state"]), row)
    return sorted(by_hash), by_hash


def q_record(
    evaluator: ArtifactEvaluator, row: dict[str, Any], state_hash: str, simulations: int
) -> dict[str, Any]:
    """Extract root-child Q directly from the production PUCT tree."""
    game = KalahGame.from_state(decode_kalah_v3_base_state(list(row["state"])))
    legal = [int(move) for move in game.possible_moves()]
    search = PUCT(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=1.25,
        rng=random.Random(state_seed(state_hash)),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    )
    visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    children = []
    for action in legal:
        child = root.children[action]
        children.append(
            {
                "action": action,
                "visits": int(visits[action]),
                "q_value": float(child.q_value),
                "prior": float(child.prior),
                "q_source": "visited_backup" if child.visit_count else "zero_fpu",
            }
        )
    return {"root_visits": int(root.visit_count), "children": children}


def _q_chunk(
    artifact: str, tasks: list[tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(Path(artifact))
    return [
        {
            "schema": "azlite_pr214_root_q_cache_v1",
            "state_hash": state_hash,
            "q384": q_record(evaluator, row, state_hash, 384),
            "q1200": q_record(evaluator, row, state_hash, 1200),
        }
        for state_hash, row in tasks
    ]


def build_q_cache(
    rows: list[dict[str, Any]], artifact: Path, path: Path, workers: int
) -> dict[str, dict[str, Any]]:
    """Resume a deterministic full-unique-state cache of production PUCT roots."""
    hashes, by_hash = unique_rows(rows)
    cached: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for entry in read_jsonl(path):
            key = entry.get("state_hash")
            if key in by_hash and entry.get("schema") == "azlite_pr214_root_q_cache_v1":
                cached[key] = entry
    missing = [key for key in hashes if key not in cached]
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        count = min(max(1, workers), len(missing))
        chunks = [
            [(key, by_hash[key]) for key in missing[offset::count]]
            for offset in range(count)
        ]
        with path.open("a", encoding="utf-8") as handle:
            with concurrent.futures.ProcessPoolExecutor(max_workers=count) as pool:
                futures = [
                    pool.submit(_q_chunk, str(artifact), chunk) for chunk in chunks
                ]
                for future in futures:
                    for entry in future.result():
                        handle.write(json.dumps(entry, sort_keys=True) + "\n")
                        cached[entry["state_hash"]] = entry
    if set(cached) != set(hashes):
        raise RuntimeError("root-Q cache does not cover every unique replay state")
    return cached


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def _legal_policy(policy: np.ndarray, legal: list[int]) -> np.ndarray:
    result = np.zeros(6, dtype=np.float64)
    result[legal] = policy[legal]
    result[legal] /= result[legal].sum()
    return result


def classification_records(
    rows: list[dict[str, Any]],
    parent: np.ndarray,
    clean: dict[int, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Freeze state-level Q advantage groups before any candidate is trained."""
    hashes, by_hash = unique_rows(rows)
    index_by_hash = {}
    for index, row in enumerate(rows):
        index_by_hash.setdefault(canonical_hash(row["state"]), index)
    records: dict[str, dict[str, Any]] = {}
    for key in hashes:
        row = by_hash[key]
        index = index_by_hash[key]
        legal = [int(value) for value in row["legal_moves"]]
        p1, stored = (
            _legal_policy(parent[index], legal),
            _legal_policy(np.asarray(row["policy"]), legal),
        )
        q_values: dict[str, np.ndarray] = {}
        for budget in (384, 1200):
            children = cache[key][f"q{budget}"]["children"]
            q = np.zeros(6, dtype=np.float64)
            for child in children:
                q[int(child["action"])] = float(child["q_value"])
            q_values[str(budget)] = q
        delta384 = float(np.dot(stored - p1, q_values["384"]))
        delta1200 = float(np.dot(stored - p1, q_values["1200"]))
        group = (
            "robust_positive"
            if delta384 > 0 and delta1200 > 0
            else "budget_conflicted"
            if (delta384 > 0) != (delta1200 > 0)
            else "robust_nonpositive"
        )
        ordered_q = {
            budget: sorted(
                legal, key=lambda action: (-q_values[budget][action], action)
            )
            for budget in q_values
        }
        stored_top = min(legal, key=lambda action: (-stored[action], action))
        p1_top = min(legal, key=lambda action: (-p1[action], action))
        clean384 = _legal_policy(np.asarray(clean[index]["clean384"]["policy"]), legal)
        clean1200 = _legal_policy(
            np.asarray(clean[index]["clean1200"]["policy"]), legal
        )
        records[key] = {
            "group": group,
            "delta_q_384": delta384,
            "delta_q_1200": delta1200,
            "player": int(row["player"]),
            "move_bucket": move_index_bucket(int(row["move_index"])),
            "legal_count": len(legal),
            "teacher_entropy": entropy(stored),
            "teacher_entropy_quartile": None,
            "stored_clean1200_top1_agree": stored_top
            == min(legal, key=lambda action: (-clean1200[action], action)),
            "expected_q": {
                budget: {
                    "parent": float(np.dot(p1, q_values[budget])),
                    "stored": float(np.dot(stored, q_values[budget])),
                    "clean384": float(np.dot(clean384, q_values[budget])),
                    "clean1200": float(np.dot(clean1200, q_values[budget])),
                }
                for budget in q_values
            },
            "best_action_q_margin": {
                budget: float(
                    q_values[budget][ordered_q[budget][0]]
                    - q_values[budget][ordered_q[budget][1]]
                )
                if len(legal) > 1
                else 0.0
                for budget in q_values
            },
            "teacher_top1_q_rank": {
                budget: ordered_q[budget].index(stored_top) + 1 for budget in q_values
            },
            "parent_top1_q_rank": {
                budget: ordered_q[budget].index(p1_top) + 1 for budget in q_values
            },
            "stored_ce_opportunity": float(
                _cross_entropy(p1[None, :], stored[None, :])[0]
            ),
        }
    edges = np.quantile(
        np.asarray([record["teacher_entropy"] for record in records.values()]),
        [0.25, 0.5, 0.75],
    )
    for record in records.values():
        record["teacher_entropy_quartile"] = (
            int(np.searchsorted(edges, record["teacher_entropy"], side="right")) + 1
        )
    return records, {
        canonical_hash(row["state"]): canonical_hash(row["state"]) for row in rows
    }


def matched_random(records: dict[str, dict[str, Any]]) -> set[str]:
    """Match the robust-positive count using only pre-Q replay covariates."""
    positive = {
        key for key, record in records.items() if record["group"] == "robust_positive"
    }
    strata: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for key, record in records.items():
        strata[
            (
                record["player"],
                record["move_bucket"],
                record["legal_count"],
                record["teacher_entropy_quartile"],
            )
        ].append(key)
    selected: set[str] = set()
    total = len(records)
    for keys in strata.values():
        size = int(round(len(positive) * len(keys) / total))
        selected.update(
            sorted(
                keys,
                key=lambda key: hashlib.sha256(
                    f"{CONTROL_SEED}:{key}".encode()
                ).hexdigest(),
            )[:size]
        )
    ordered = sorted(
        records,
        key=lambda key: hashlib.sha256(
            f"{CONTROL_SEED}:fill:{key}".encode()
        ).hexdigest(),
    )
    if len(selected) < len(positive):
        selected.update(key for key in ordered if key not in selected)
    return set(
        sorted(
            selected,
            key=lambda key: hashlib.sha256(
                f"{CONTROL_SEED}:trim:{key}".encode()
            ).hexdigest(),
        )[: len(positive)]
    )


def lane_target(
    teacher: torch.Tensor,
    parent: torch.Tensor,
    mask: torch.Tensor,
    selected: np.ndarray,
) -> torch.Tensor:
    selected_tensor = torch.as_tensor(selected, dtype=torch.bool, device=teacher.device)
    mixed = 0.05 * teacher + 0.95 * parent
    target = torch.where(selected_tensor[:, None], mixed, parent)
    target = torch.where(mask.bool(), target, torch.zeros_like(target))
    return target / target.sum(dim=1, keepdim=True)


def train_lane(
    lane: str,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    parent_state: dict[str, torch.Tensor],
    selected: set[str],
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
    model.train()
    for step, indexes in enumerate(plan[:STEPS], 1):
        batch_rows = [
            rows[int(source[int(index)])] for index in indexes if int(index) >= 0
        ]
        batch = _batch(batch_rows, np.arange(len(batch_rows)), device)
        parent_policy = incumbent_policy_batch(parent, batch)
        flags = np.asarray(
            [canonical_hash(row["state"]) in selected for row in batch_rows]
        )
        target = lane_target(batch["p"], parent_policy, batch["mask"], flags)
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


def drift(
    candidate: np.ndarray, parent: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    l1 = np.abs(candidate - parent).sum(axis=1)
    midpoint = np.maximum((candidate + parent) / 2, 1e-12)
    return {
        "mean_l1": float(l1.mean()),
        "p90_l1": float(np.percentile(l1, 90)),
        "p95_l1": float(np.percentile(l1, 95)),
        "p99_l1": float(np.percentile(l1, 99)),
        "max_l1": float(l1.max()),
        "js": float(
            np.mean(
                0.5
                * (
                    np.sum(
                        candidate * np.log(np.maximum(candidate, 1e-12) / midpoint),
                        axis=1,
                    )
                    + np.sum(
                        parent * np.log(np.maximum(parent, 1e-12) / midpoint), axis=1
                    )
                )
            )
        ),
        "top1_disagreement": float(
            np.mean(
                np.argmax(np.where(mask, candidate, -np.inf), axis=1)
                != np.argmax(np.where(mask, parent, -np.inf), axis=1)
            )
        ),
    }


def metrics(
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    parent_state: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    selected: set[str],
    pure_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    parent = output(parent_state, x, mask)
    stored = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    target = np.where(
        np.asarray([canonical_hash(row["state"]) in selected for row in rows])[:, None],
        0.05 * stored + 0.95 * parent,
        parent,
    )
    baseline_ce = float(np.mean(_cross_entropy(parent, stored)))
    pure_ce = float(np.mean(_cross_entropy(output(pure_state, x, mask), stored)))
    groups = np.asarray(
        [records[canonical_hash(row["state"])]["group"] for row in rows]
    )
    result: dict[str, Any] = {}
    for step, (state, _) in snapshots.items():
        candidate = output(state, x, mask)
        candidate_ce = float(np.mean(_cross_entropy(candidate, stored)))
        result[str(step)] = {
            "ce_candidate_stored384": candidate_ce,
            "ce_candidate_p1": float(np.mean(_cross_entropy(candidate, parent))),
            "ce_candidate_lane_target": float(
                np.mean(_cross_entropy(candidate, target))
            ),
            "search_target_ce_improvement_vs_p1": baseline_ce - candidate_ce,
            "fit_fraction": None
            if step == 0
            else (baseline_ce - candidate_ce) / (baseline_ce - pure_ce),
            "fit_by_group": {
                group: float(
                    np.mean(
                        _cross_entropy(parent[groups == group], stored[groups == group])
                    )
                    - np.mean(
                        _cross_entropy(
                            candidate[groups == group], stored[groups == group]
                        )
                    )
                )
                for group in GROUPS
            },
            "drift": drift(candidate, parent, mask),
        }
    return result


def value_support(
    state: dict[str, torch.Tensor],
    parent_state: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    hashes, unique = unique_rows(rows)
    x = np.asarray([unique[key]["state"] for key in hashes], np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    candidate, parent = output(state, x, mask), output(parent_state, x, mask)
    result = {}
    groups = np.asarray([records[key]["group"] for key in hashes])
    for budget in (384, 1200):
        values = []
        for index, key in enumerate(hashes):
            q = np.zeros(6)
            for child in cache[key][f"q{budget}"]["children"]:
                q[int(child["action"])] = child["q_value"]
            values.append(float(np.dot(candidate[index] - parent[index], q)))
        values = np.asarray(values)
        result[str(budget)] = {
            **percentile_summary(values),
            "fraction_positive": float(np.mean(values > 0)),
            "robust_positive": percentile_summary(values[groups == "robust_positive"]),
            "other": percentile_summary(values[groups != "robust_positive"]),
        }
    return result


def audit(
    records: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        values384 = np.asarray([item["delta_q_384"] for item in items])
        values1200 = np.asarray([item["delta_q_1200"] for item in items])
        return {
            "count": len(items),
            "delta_q_384": percentile_summary(values384),
            "delta_q_1200": percentile_summary(values1200),
            "sign_agreement": float(np.mean((values384 > 0) == (values1200 > 0))),
            "stored384_ce_opportunity": float(
                sum(item["stored_ce_opportunity"] for item in items)
            ),
        }

    all_items = list(records.values())
    total_opportunity = sum(item["stored_ce_opportunity"] for item in all_items)
    groups = {
        group: summarize([item for item in all_items if item["group"] == group])
        for group in GROUPS
    }
    for details in groups.values():
        details["stored384_ce_opportunity_fraction"] = (
            details["stored384_ce_opportunity"] / total_opportunity
        )
    row_groups = [records[canonical_hash(row["state"])]["group"] for row in rows]
    split_keys = (
        "player",
        "move_bucket",
        "legal_count",
        "teacher_entropy_quartile",
        "stored_clean1200_top1_agree",
    )
    root_action_value = {
        budget: {
            "expected_q": {
                policy: percentile_summary(
                    np.asarray(
                        [item["expected_q"][budget][policy] for item in all_items]
                    )
                )
                for policy in ("parent", "stored", "clean384", "clean1200")
            },
            "best_action_q_margin": percentile_summary(
                np.asarray([item["best_action_q_margin"][budget] for item in all_items])
            ),
            "teacher_top1_q_rank": percentile_summary(
                np.asarray([item["teacher_top1_q_rank"][budget] for item in all_items])
            ),
            "parent_top1_q_rank": percentile_summary(
                np.asarray([item["parent_top1_q_rank"][budget] for item in all_items])
            ),
        }
        for budget in ("384", "1200")
    }
    return {
        "overall": summarize(all_items),
        "groups": groups,
        "unique_group_fraction": {
            group: len([item for item in all_items if item["group"] == group])
            / len(all_items)
            for group in GROUPS
        },
        "replay_row_group_fraction": {
            group: row_groups.count(group) / len(row_groups) for group in GROUPS
        },
        "splits": {
            key: {
                str(value): summarize(
                    [item for item in all_items if item[key] == value]
                )
                for value in sorted({item[key] for item in all_items}, key=str)
            }
            for key in split_keys
        },
        "root_action_value": root_action_value,
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# PR #214 Root-Q Advantage-Gated Adapter Retrain",
        "",
        f"**Primary classification:** `{summary['classification']}`",
        "",
        f"**Recommended next experiment:** {summary['recommended_next_experiment']}",
        "",
        "## Root-Q Audit",
        "",
        "Child Q is backed up in the root player-to-move perspective. Every legal child is cached; an unvisited child records the existing `zero_fpu` Q=0 contract explicitly.",
        "",
        "| Group | Unique fraction | Replay-row fraction | Stored CE opportunity |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group in GROUPS:
        details = summary["audit"]["groups"][group]
        lines.append(
            f"| {group} | {summary['audit']['unique_group_fraction'][group]:.4f} | {summary['audit']['replay_row_group_fraction'][group]:.4f} | {details['stored384_ce_opportunity_fraction']:.4f} |"
        )
    lines += [
        "",
        "| Delta-Q budget | Mean | Median | P10 | P25 | P75 | P90 | P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in (384, 1200):
        values = summary["audit"]["overall"][f"delta_q_{budget}"]
        lines.append(
            f"| {budget} | {values['mean']:.6f} | {values['median']:.6f} | "
            f"{values['p10']:.6f} | {values['p25']:.6f} | {values['p75']:.6f} | "
            f"{values['p90']:.6f} | {values['p95']:.6f} |"
        )
    lines += [
        "",
        "## Training",
        "",
        "| Lane | Step | CE(stored384) | CE(P1) | CE(lane target) | Fit fraction | Mean L1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in LANES:
        for step in CHECKPOINTS:
            metric = summary["lanes"][lane]["metrics"][str(step)]
            lines.append(
                f"| {lane} | {step} | {metric['ce_candidate_stored384']:.6f} | {metric['ce_candidate_p1']:.6f} | {metric['ce_candidate_lane_target']:.6f} | {metric['fit_fraction']:.4f} | {metric['drift']['mean_l1']:.6f} |"
            )
    lines += [
        "",
        "| Lane | Step | Robust-positive fit | Budget-conflicted fit | Robust-nonpositive fit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for lane in LANES:
        for step in CHECKPOINTS:
            fit = summary["lanes"][lane]["metrics"][str(step)]["fit_by_group"]
            lines.append(
                f"| {lane} | {step} | {fit['robust_positive']:.6f} | "
                f"{fit['budget_conflicted']:.6f} | {fit['robust_nonpositive']:.6f} |"
            )
    lines += [
        "",
        "## Value-Support Diagnostics",
        "",
        "| Lane | Step | Q budget | Mean expected-Q change | Positive fraction |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for lane in LANES:
        for step in CHECKPOINTS:
            for budget in ("384", "1200"):
                support = summary["lanes"][lane]["value_support"][str(step)][budget]
                lines.append(
                    f"| {lane} | {step} | {budget} | {support['mean']:+.6f} | "
                    f"{support['fraction_positive']:.4f} |"
                )
    lines += [
        "",
        "## Arena",
        "",
        "| Lane | Step | Context | Effect | 95% CI | Seat P0 | Seat P1 |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for lane, checkpoints in summary["arena"].items():
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
                "hashes": summary["hashes"],
                "invariants": summary["invariants"],
                "search_contract": summary["search_contract"],
                "matched_random": summary["matched_random"],
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
        "--workdir", type=Path, default=Path("/tmp/azlite_pr214_root_q_advantage")
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
        "--teacher-workdir",
        type=Path,
        default=Path("/tmp/azlite_pr214_teacher_target_retrain"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-root-q-advantage-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-root-q-advantage-results.md",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    determinism = configure_determinism(device, SEED)
    p1_checkpoint = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    p1_artifact = p1_checkpoint.parent / "artifact"
    p0_artifact = REPO_ROOT / "model-artifact/current"
    a16_checkpoint = args.adapter_workdir / "artifacts/step_0016/checkpoint.npz"
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    manifest = verify_manifest(args.adapter_workdir / "training_manifest.json")
    rows = read_jsonl(replay)
    parent = new_model(device)
    load_checkpoint_into_model(parent, p1_checkpoint)
    a16 = new_model(device)
    load_checkpoint_into_model(a16, a16_checkpoint)
    parent_state = {
        name: value.detach().cpu().clone()
        for name, value in parent.state_dict().items()
    }
    paths = {key: Path(value) for key, value in manifest["artifact_paths"].items()}
    plan, source = (
        np.load(paths["batch_indexes"], allow_pickle=False),
        np.load(paths["train_source_indexes"], allow_pickle=False),
    )
    invariants = {
        "p1_checkpoint_hash": sha256_file(p1_checkpoint) == P1_CHECKPOINT_SHA,
        "a16_state_hash": model_state_hash(a16.state_dict()) == A16_STATE_SHA,
        "replay_hash": sha256_file(replay) == REPLAY_SHA,
        "pr217_batch_manifest_hash": sha256_file(
            args.adapter_workdir / "training_manifest.json"
        )
        == PR217_BATCH_MANIFEST_SHA,
        "pr217_clean_target_cache_hash": sha256_file(
            args.teacher_workdir / "clean_p1_targets.jsonl"
        )
        == PR217_CLEAN_TARGET_CACHE_SHA,
        "pr214_batch_plan": bool(
            len(plan) == STEPS
            and np.all((plan == -1) | ((plan >= 0) & (plan < len(source))))
        ),
        "full_replay_state_round_trip": all(
            state_round_trips_kalah_v3(row["state"]) for row in rows
        ),
    }
    verify_puct_q_semantics()
    invariants["root_q_perspective"] = True
    if not all(invariants.values()):
        raise RuntimeError(f"immutable contract failed: {invariants}")
    q_cache = build_q_cache(
        rows, p1_artifact, args.workdir / "root_q_cache.jsonl", args.workers
    )
    clean = build_clean_target_cache(
        rows, p1_artifact, args.teacher_workdir / "clean_p1_targets.jsonl", args.workers
    )
    x = np.asarray([row["state"] for row in rows], np.float32)
    parent_policy = output(parent_state, x, legal_mask_matrix_for_encoded_states(x))
    records, _unused = classification_records(rows, parent_policy, clean, q_cache)
    random_selected = matched_random(records)
    robust_selected = {
        key for key, record in records.items() if record["group"] == "robust_positive"
    }
    selections = {
        "baseline_stored384": set(records),
        "robust_advantage": robust_selected,
        "matched_random": random_selected,
    }
    snapshots = {
        lane: train_lane(
            lane, rows, manifest, parent_state, selected, args.workdir / lane, device
        )
        for lane, selected in selections.items()
    }
    invariants.update(
        {
            "same_initial_state": all(
                model_state_hash(snapshots[lane][0][0])
                == model_state_hash(parent_state)
                for lane in LANES
            ),
            "non_adapter_parameters_bit_identical": all(
                torch.equal(state[name], parent_state[name])
                for lane in LANES
                for state, _ in snapshots[lane].values()
                for name in state
                if name not in ADAPTER_KEYS
            ),
            "matched_random_unique_count": len(random_selected) == len(robust_selected),
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
                str(step): model_state_hash(state)
                for step, (state, _) in snapshots[lane].items()
            },
            "metrics": metrics(
                snapshots[lane],
                parent_state,
                rows,
                records,
                selections[lane],
                pure_state,
            ),
            "value_support": {
                str(step): value_support(state, parent_state, rows, records, q_cache)
                for step, (state, _) in snapshots[lane].items()
                if step in CHECKPOINTS
            },
        }
        for lane in LANES
    }
    suite, suite_hash = _suite()
    arena: dict[str, Any] = defaultdict(dict)
    control_cache: dict[str, list[dict[str, Any]]] = {}
    requested = {
        "baseline_stored384": {46: ("384:256", "1200:1200")},
        "robust_advantage": {16: ("384:256",), 46: ("384:256",)},
        "matched_random": {16: ("384:256",), 46: ("384:256",)},
    }
    for lane, steps in requested.items():
        for step, contexts in steps.items():
            artifact = export(
                snapshots[lane][step][0],
                args.workdir / lane / f"artifact_step_{step:04d}",
                f"pr214_{lane}_step_{step}",
            )
            for context in contexts:
                control = control_cache.setdefault(
                    context,
                    arena_records(
                        workdir=args.workdir / "arena_control",
                        challenger=p1_artifact,
                        current=p1_artifact,
                        context=context,
                        role="p1_control",
                        workers=args.workers,
                        suite_hash=suite_hash,
                    ),
                )
                result = paired_opening_candidate_effect(
                    arena_records(
                        workdir=args.workdir / "arena",
                        challenger=artifact,
                        current=p1_artifact,
                        context=context,
                        role=f"{lane}_{step}",
                        workers=args.workers,
                        suite_hash=suite_hash,
                    ),
                    control,
                )
                arena[lane].setdefault(str(step), {})[context] = {
                    key: result[key]
                    for key in (
                        "paired_candidate_effect",
                        "opening_bootstrap_ci",
                        "p0_effect",
                        "p1_effect",
                    )
                }
    for lane in ("robust_advantage", "matched_random"):
        for step in (16, 46):
            result = arena[lane][str(step)]["384:256"]
            safe = (
                result["opening_bootstrap_ci"]["lower_95"] >= 0
                and lane_results[lane]["metrics"][str(step)]["fit_fraction"] >= 0.25
            )
            if safe:
                artifact = export(
                    snapshots[lane][step][0],
                    args.workdir / lane / f"artifact_step_{step:04d}",
                    f"pr214_{lane}_step_{step}",
                )
                control = control_cache.setdefault(
                    "1200:1200",
                    arena_records(
                        workdir=args.workdir / "arena_control",
                        challenger=p1_artifact,
                        current=p1_artifact,
                        context="1200:1200",
                        role="p1_control",
                        workers=args.workers,
                        suite_hash=suite_hash,
                    ),
                )
                result = paired_opening_candidate_effect(
                    arena_records(
                        workdir=args.workdir / "arena",
                        challenger=artifact,
                        current=p1_artifact,
                        context="1200:1200",
                        role=f"{lane}_{step}",
                        workers=args.workers,
                        suite_hash=suite_hash,
                    ),
                    control,
                )
                arena[lane][str(step)]["1200:1200"] = {
                    key: result[key]
                    for key in (
                        "paired_candidate_effect",
                        "opening_bootstrap_ci",
                        "p0_effect",
                        "p1_effect",
                    )
                }
    for step, contexts in arena.get("robust_advantage", {}).items():
        if {"384:256", "1200:1200"}.issubset(contexts) and all(
            value["opening_bootstrap_ci"]["lower_95"] >= 0
            for value in contexts.values()
        ):
            artifact = export(
                snapshots["robust_advantage"][int(step)][0],
                args.workdir / "robust_advantage" / f"artifact_step_{int(step):04d}",
                f"pr214_robust_advantage_step_{step}",
            )
            for context in ("384:256", "1200:1200"):
                control = arena_records(
                    workdir=args.workdir / "arena_p0_control",
                    challenger=p0_artifact,
                    current=p0_artifact,
                    context=context,
                    role="p0_control",
                    workers=args.workers,
                    suite_hash=suite_hash,
                )
                result = paired_opening_candidate_effect(
                    arena_records(
                        workdir=args.workdir / "arena_p0",
                        challenger=artifact,
                        current=p0_artifact,
                        context=context,
                        role=f"robust_advantage_{step}",
                        workers=args.workers,
                        suite_hash=suite_hash,
                    ),
                    control,
                )
                arena["robust_advantage"][step][f"vs_p0_{context}"] = {
                    key: result[key]
                    for key in (
                        "paired_candidate_effect",
                        "opening_bootstrap_ci",
                        "p0_effect",
                        "p1_effect",
                    )
                }
    baseline = arena["baseline_stored384"]["46"]
    reproduction = (
        baseline["384:256"]["paired_candidate_effect"] == -0.01953125
        and baseline["1200:1200"]["paired_candidate_effect"] == -0.0234375
    )
    invariants["baseline_reproduced"] = reproduction
    invariants["all_passed"] = all(invariants.values())
    robust = arena.get("robust_advantage", {}).get("46", {})
    random46 = arena.get("matched_random", {}).get("46", {})
    robust_p1_safe = all(
        value["opening_bootstrap_ci"]["lower_95"] >= 0
        for name, value in robust.items()
        if not name.startswith("vs_p0_")
    )
    robust_fit = (
        lane_results["robust_advantage"]["metrics"]["46"]["fit_fraction"] >= 0.25
    )
    p0_results = [value for name, value in robust.items() if name.startswith("vs_p0_")]
    robust_p0_safe = not p0_results or all(
        value["opening_bootstrap_ci"]["lower_95"] >= 0 for value in p0_results
    )
    more_value_supported = all(
        lane_results["robust_advantage"]["value_support"]["46"][budget]["mean"]
        > lane_results["matched_random"]["value_support"]["46"][budget]["mean"]
        for budget in ("384", "1200")
    )
    if not invariants["all_passed"]:
        classification, next_experiment = (
            "invariant_failure",
            "Repair the immutable-input, Q-perspective, baseline, or matched-control contract.",
        )
    elif (
        robust_p1_safe
        and robust_fit
        and robust_p0_safe
        and more_value_supported
        and "1200:1200" in robust
        and robust["1200:1200"]["paired_candidate_effect"] >= 0
        and random46.get("384:256", {}).get("paired_candidate_effect", 0)
        < robust["384:256"]["paired_candidate_effect"]
    ):
        classification, next_experiment = (
            "value_supported_teacher_rescues_update",
            "Generate a fresh AlphaZero iteration using the same value-supported policy target gate prospectively.",
        )
    elif robust_p1_safe and not robust_fit:
        classification, next_experiment = (
            "advantage_filter_safe_but_weak",
            "Increase search reliability or construct a soft advantage-weighted target.",
        )
    elif (
        robust_p1_safe
        and robust_fit
        and random46.get("384:256", {}).get("paired_candidate_effect", -1)
        >= robust["384:256"]["paired_candidate_effect"] - 0.01
    ):
        classification, next_experiment = (
            "generic_supervision_reduction_explains_result",
            "Do not add a Q-based state selector; test a simpler update-volume control.",
        )
    elif (
        lane_results["robust_advantage"]["metrics"]["46"]["fit_fraction"] >= 0.25
        and robust.get("384:256", {}).get("paired_candidate_effect", 0) < 0
    ):
        classification, next_experiment = (
            "root_q_sign_not_predictive",
            "Move from state-level expected advantage to action-level/search-trajectory sensitivity rather than another teacher-budget change.",
        )
    else:
        classification, next_experiment = (
            "inconclusive",
            "Retain the frozen mask and inspect the preregistered diagnostics without adding lanes.",
        )
    summary = {
        "schema": "azlite_pr214_root_q_advantage_v1",
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
            "batch_manifest": sha256_file(
                args.adapter_workdir / "training_manifest.json"
            ),
            "root_q_cache": sha256_file(args.workdir / "root_q_cache.jsonl"),
            "clean_target_cache": sha256_file(
                args.teacher_workdir / "clean_p1_targets.jsonl"
            ),
            "arena_suite": suite_hash,
        },
        "search_contract": {
            "simulations": [384, 1200],
            "c_puct": 1.25,
            "root_noise": False,
            "fpu_mode": "zero",
            "unvisited_q_rule": "zero_fpu: q_value=0.0, recorded explicitly",
            "seed": "sha256(pr214-teacher-audit:encoded-state-hash)",
            "q_perspective": "root player to move; higher is better",
        },
        "determinism": determinism,
        "invariants": invariants,
        "audit": audit(records, rows),
        "matched_random": {
            "seed": CONTROL_SEED,
            "robust_positive_unique_states": len(robust_selected),
            "matched_random_unique_states": len(random_selected),
            "robust_mask_sha256": hashlib.sha256(
                "".join(sorted(robust_selected)).encode()
            ).hexdigest(),
            "random_mask_sha256": hashlib.sha256(
                "".join(sorted(random_selected)).encode()
            ).hexdigest(),
        },
        "lanes": lane_results,
        "arena": dict(arena),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_report.write_text(render(summary), encoding="utf-8")
    print(classification)


if __name__ == "__main__":
    main()
