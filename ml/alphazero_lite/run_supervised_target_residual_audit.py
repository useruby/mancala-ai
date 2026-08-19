#!/usr/bin/env python3
# ruff: noqa: E402
"""Supervised target-residual quality and causal-alignment audit.

Determines whether the reproducibly harmful first supervised update from PR #197
is caused by policy replay targets that point away from better moves, by
noisy/off-policy terminal value targets, by both, or by targets that are
individually sensible but become harmful when optimized through the current
policy/value objective.

No training. No optimizer steps that mutate a candidate. No new self-play.
No target transformation. Every diagnostic gradient and every virtual Adam step
begins from the frozen PR #191 initialization and is discarded afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_seed_contract import (
    SEED_CONTRACT_VERSION,
    derive_search_seed,
    stable_hash,
    stable_seed,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_aggregate_gradient_stability_audit import (
    AGGREGATE_REPLICATES,
    EFFECTIVE_BATCH_SIZES,
    aggregate_microbatches,
    aggregate_update,
    group_flat,
    mean_full_gradient,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_distribution_aligned_selfplay_iteration import _decoded_state
from ml.alphazero_lite.run_game_shard_gradient_stability_audit import (
    CURRENT_HASH,
    GROUPS,
    SHARDS,
    cosine,
    deterministic_batches,
    fresh_state,
    new_model,
    parameter_group,
    partition,
    phase,
    vectors,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch
from ml.alphazero_lite.run_policy_target_noise_causal_closeout import (
    continuation_seed_identity,
    forced_continuation,
)
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (
    _visit_policy,
    softmax_masked,
)
from ml.alphazero_lite.self_play import (
    CheckpointEvaluator,
    PUCT,
    build_eval_search_options,
    encode_state,
)
from ml.alphazero_lite.train import (
    PolicyValueNet,
    legal_mask_matrix_for_encoded_states,
)

PROBE_SIZE = 1024
PROBE_SCHEMA = "azlite_target_residual_probe_v1"
SEARCH_C_PUCT = 1.25
PHASE_C_CONTINUATION_BUDGETS = (768, 1200)
PHASE_C_SEARCH_BUDGETS = (384, 1200)
PHASE_D_CONTINUATION_BUDGET = 1200
PHASE_D_SUBSET = 256
PHASE_E_CONTINUATION_BUDGETS = (768, 1200)
NAMESPACE = "azlite_supervised_target_residual_audit_v1"
CONTINUATION_NAMESPACE = "azlite_target_residual_continuation_v1"
BOOTSTRAP_SAMPLES = 10_000


def kl_divergence(target: np.ndarray, current: np.ndarray) -> float:
    """KL(target || current) over the support of target, in nats."""
    legal = target > 0.0
    if not np.any(legal):
        return 0.0
    t = np.clip(target[legal], 1e-12, None)
    c = np.clip(current[legal], 1e-12, None)
    return float(np.sum(t * np.log(t / c)))


def policy_entropy(policy: np.ndarray) -> float:
    legal = policy > 0.0
    if not np.any(legal):
        return 0.0
    p = policy[legal]
    return float(-np.sum(p * np.log(p)))


def top_move(policy: np.ndarray, legal_moves: list[int]) -> int:
    legal = [int(move) for move in legal_moves]
    return max(legal, key=lambda move: (float(policy[move]), -move))


def bootstrap_median_ci(values: list[float], *, seed: int) -> dict[str, Any]:
    """Bootstrap the median estimator itself, not the mean."""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.median(
        arr[rng.integers(0, len(arr), size=(BOOTSTRAP_SAMPLES, len(arr)))], axis=1
    )
    return {
        "estimator": "median",
        "median": float(np.median(arr)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def bootstrap_mean_ci(values: list[float], *, seed: int) -> dict[str, Any]:
    """Bootstrap the mean estimator and label it as such."""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, len(arr), size=(BOOTSTRAP_SAMPLES, len(arr)))].mean(
        axis=1
    )
    return {
        "estimator": "mean",
        "mean": float(arr.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def bootstrap_difference_ci(
    left: list[float], right: list[float], *, seed: int, statistic: str = "mean"
) -> dict[str, Any]:
    """Direct bootstrap of ``left - right`` without marginal-CI inference."""
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    left_draws = left_arr[
        rng.integers(0, len(left_arr), size=(BOOTSTRAP_SAMPLES, len(left_arr)))
    ]
    right_draws = right_arr[
        rng.integers(0, len(right_arr), size=(BOOTSTRAP_SAMPLES, len(right_arr)))
    ]
    if statistic == "mean":
        draws = left_draws.mean(axis=1) - right_draws.mean(axis=1)
        point = float(left_arr.mean() - right_arr.mean())
    else:
        draws = np.median(left_draws, axis=1) - np.median(right_draws, axis=1)
        point = float(np.median(left_arr) - np.median(right_arr))
    return {
        "statistic": statistic,
        "point": point,
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def bootstrap_paired_ci(values: list[float], *, seed: int) -> dict[str, Any]:
    """Cluster-bootstrap a list of paired per-state deltas by unique state."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "lower_95": 0.0,
            "upper_95": 0.0,
        }
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, len(arr), size=(BOOTSTRAP_SAMPLES, len(arr)))].mean(
        axis=1
    )
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def current_policy_value(
    model: PolicyValueNet, states: np.ndarray, mask: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return the current legal policy and value prediction for a state matrix."""
    model.eval()
    policies: list[np.ndarray] = []
    values: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in np.array_split(
            np.arange(states.shape[0]), math.ceil(states.shape[0] / 2048)
        ):
            if chunk.size == 0:
                continue
            logits, value = model(torch.as_tensor(states[chunk], device=device))
            policies.append(
                softmax_masked(logits.cpu().numpy(), mask[chunk].astype(bool))
            )
            values.append(value.cpu().numpy().reshape(-1))
    return np.concatenate(policies), np.concatenate(values)


def probe_row_valid(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    try:
        state = _decoded_state(row)
        game = KalahGame.from_state(state)
        encoded = np.asarray(
            encode_state(state, input_encoding="kalah_v3"), dtype=np.float32
        )
        persisted = np.asarray(row["state"], dtype=np.float32)
        if encoded.shape != persisted.shape or not np.array_equal(encoded, persisted):
            return None, "re-encode mismatch"
        if not game.possible_moves():
            return None, "terminal state"
    except (KeyError, TypeError, ValueError) as exc:
        return None, str(exc)
    return state, ""


def build_probe(
    rows: list[dict[str, Any]],
    source_indexes: np.ndarray,
    current_policy: np.ndarray,
    current_value: np.ndarray,
    *,
    size: int = PROBE_SIZE,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select exactly ``size`` unique training states stratified by residual."""
    n = len(rows)
    policy_residual = np.zeros(n, dtype=np.float64)
    value_residual = np.zeros(n, dtype=np.float64)
    entropy = np.zeros(n, dtype=np.float64)
    phase_arr = np.empty(n, dtype=object)
    player_arr = np.zeros(n, dtype=np.int64)
    hashes: list[str] = []
    decoded: list[dict[str, Any] | None] = [None] * n
    move_index = np.zeros(n, dtype=np.int64)
    noise_enabled = np.zeros(n, dtype=bool)
    for i in range(n):
        state, reason = probe_row_valid(rows[i])
        decoded[i] = state
        if state is None:
            policy_residual[i] = np.inf
            value_residual[i] = np.inf
            phase_arr[i] = ""
            hashes.append("")
            continue
        policy_residual[i] = kl_divergence(
            np.asarray(rows[i]["policy"]), current_policy[i]
        )
        value_residual[i] = abs(float(rows[i]["value"]) - float(current_value[i]))
        entropy[i] = policy_entropy(np.asarray(rows[i]["policy"]))
        phase_arr[i] = phase(rows[i])
        player_arr[i] = int(rows[i]["player"])
        hashes.append(stable_hash(state))
        move_index[i] = int(rows[i].get("move_index", 0))
        noise_enabled[i] = bool(rows[i].get("action_sampling_noise_enabled", False))

    valid = np.asarray([hash_ != "" for hash_ in hashes], dtype=bool)
    if int(valid.sum()) < size:
        raise RuntimeError(f"only {int(valid.sum())} valid unique-ish training rows")

    p_bounds = np.quantile(policy_residual[valid], [0.25, 0.5, 0.75])
    v_bounds = np.quantile(value_residual[valid], [0.25, 0.5, 0.75])
    q_p = np.full(n, -1, dtype=np.int64)
    q_v = np.full(n, -1, dtype=np.int64)
    q_p[valid] = np.digitize(policy_residual[valid], p_bounds)
    q_v[valid] = np.digitize(value_residual[valid], v_bounds)

    used: set[str] = set()
    chosen: list[int] = []

    def take_evenly(candidates: list[int], k: int) -> list[int]:
        if k <= 0 or not candidates:
            return []
        ordered = sorted(candidates, key=lambda idx: entropy[idx])
        positions = np.linspace(0, len(ordered) - 1, k).astype(int)
        out: list[int] = []
        for pos in positions:
            found = None
            for j in range(pos, len(ordered)):
                if hashes[ordered[j]] not in used:
                    found = ordered[j]
                    break
            if found is None:
                for j in range(pos, -1, -1):
                    if hashes[ordered[j]] not in used:
                        found = ordered[j]
                        break
            if found is not None:
                used.add(hashes[found])
                out.append(found)
        return out

    def take_any(candidates: list[int], k: int) -> list[int]:
        out: list[int] = []
        for idx in sorted(candidates, key=lambda idx: entropy[idx]):
            if len(out) >= k:
                break
            if hashes[idx] in used:
                continue
            used.add(hashes[idx])
            out.append(idx)
        return out

    cell_quota = size // 16
    subcell_order = [
        ("opening", 0),
        ("opening", 1),
        ("mid", 0),
        ("mid", 1),
        ("late", 0),
        ("late", 1),
    ]
    base = cell_quota // len(subcell_order)
    remainder = cell_quota % len(subcell_order)
    stratum_counts: dict[str, int] = {}
    for qp in range(4):
        for qv in range(4):
            cell = [
                i for i in range(n) if q_p[i] == qp and q_v[i] == qv and hashes[i] != ""
            ]
            cell.sort(key=lambda idx: entropy[idx])
            chosen_in_cell: list[int] = []
            for offset, (ph, pl) in enumerate(subcell_order):
                subcell = [
                    i for i in cell if phase_arr[i] == ph and player_arr[i] == pl
                ]
                quota = base + (1 if offset < remainder else 0)
                chosen_in_cell.extend(take_evenly(subcell, quota))
            need = cell_quota - len(chosen_in_cell)
            if need > 0:
                remaining = [i for i in cell if hashes[i] not in used]
                chosen_in_cell.extend(take_any(remaining, need))
            chosen.extend(chosen_in_cell)
            stratum_counts[f"p{qp}_v{qv}"] = len(chosen_in_cell)

    if len(chosen) < size:
        remaining = [i for i in range(n) if hashes[i] != "" and hashes[i] not in used]
        chosen.extend(take_any(remaining, size - len(chosen)))

    chosen = chosen[:size]
    assert len(chosen) == size, f"selected {len(chosen)} != {size}"
    assert len({hashes[i] for i in chosen}) == size, "probe states are not unique"

    probe: list[dict[str, Any]] = []
    for manifest_index, source_position in enumerate(chosen):
        row = rows[source_position]
        state = decoded[source_position]
        assert state is not None
        legal_moves = [
            int(move) for move in KalahGame.from_state(state).possible_moves()
        ]
        replay_policy = np.asarray(row["policy"], dtype=np.float64)
        cur_policy = np.asarray(current_policy[source_position], dtype=np.float64)
        probe.append(
            {
                "manifest_index": manifest_index,
                "source_index": int(source_indexes[source_position]),
                "training_position": int(source_position),
                "state_hash": hashes[source_position],
                "state": state,
                "player": int(player_arr[source_position]),
                "phase": str(phase_arr[source_position]),
                "move_index": int(move_index[source_position]),
                "action_sampling_noise_enabled": bool(noise_enabled[source_position]),
                "game_index": int(row.get("game_index", -1)),
                "trajectory_hash": str(row.get("trajectory_hash", "")),
                "legal_moves": legal_moves,
                "replay_policy": [float(value) for value in row["policy"]],
                "replay_value": float(row["value"]),
                "replay_top_move": top_move(replay_policy, legal_moves),
                "current_policy": [float(value) for value in cur_policy],
                "current_value": float(current_value[source_position]),
                "current_raw_top_move": top_move(cur_policy, legal_moves),
                "policy_target_entropy": float(entropy[source_position]),
                "policy_residual_kl": float(policy_residual[source_position]),
                "value_residual_abs": float(value_residual[source_position]),
                "policy_residual_quartile": int(q_p[source_position]),
                "value_residual_quartile": int(q_v[source_position]),
            }
        )

    manifest = {
        "schema": PROBE_SCHEMA,
        "selection": (
            "training source-index order within deterministic stratified cells; "
            "16 policy-residual/value-residual quartile cells x 64, sub-stratified by "
            "phase/player and spread across policy-target entropy"
        ),
        "state_count": len(probe),
        "unique_state_count": len({row["state_hash"] for row in probe}),
        "source_indexes": [row["source_index"] for row in probe],
        "state_hashes": [row["state_hash"] for row in probe],
        "policy_residual_quartile_boundaries": [float(value) for value in p_bounds],
        "value_residual_quartile_boundaries": [float(value) for value in v_bounds],
        "stratum_counts": stratum_counts,
        "seed": seed,
    }
    return probe, manifest


def search_seed_for_state(
    record: dict[str, Any], manifest_hash: str
) -> tuple[int, str]:
    return derive_search_seed(
        base_seed=191,
        suite_sha256=manifest_hash,
        opening_index=int(record["manifest_index"]),
        opening_state_hash=str(record["state_hash"]),
        challenger_player=int(record["player"]),
        game_within_opening=0,
        ply=0,
        canonical_current_state_hash=str(record["state_hash"]),
        acting_role="challenger",
        rng_stream_name=NAMESPACE,
        contract_version=SEED_CONTRACT_VERSION,
    )


def run_top_move_search(
    record: dict[str, Any],
    evaluator: arena.ArtifactEvaluator,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    result = arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=record["state"],
        simulations=simulations,
        seed=seed,
        c_puct=SEARCH_C_PUCT,
        search_options=build_eval_search_options(
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
            normalize_values=False,
        ),
    )
    child_stats = list(result["child_stats"])
    selected = int(result["selected_move"])
    visit_policy = _visit_policy(child_stats)
    return {
        "state_hash": record["state_hash"],
        "simulations": simulations,
        "selected_move": selected,
        "visit_policy": [float(value) for value in visit_policy],
        "root_value": float(result.get("search_root_value", result["value"])),
        "child_stats": [
            {
                "move": int(item["move"]),
                "visits": int(item["visits"]),
                "q_value": float(item["q_value"]),
            }
            for item in child_stats
        ],
    }


def _worker_initializer(checkpoint: str) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = CheckpointEvaluator(Path(checkpoint), input_encoding="kalah_v3")


_WORKER_EVALUATOR: CheckpointEvaluator | None = None


def _deterministic_continuation(task: dict[str, Any]) -> dict[str, Any]:
    """Deterministic self-play continuation from the original state (no forced move)."""
    assert _WORKER_EVALUATOR is not None
    game = KalahGame.from_state(task["state"])
    root_player = int(game.current_player)
    base_seed, seed_context_hash = continuation_seed_identity(
        state_hash=task["state_hash"],
        continuation_budget=task["continuation_budget"],
        root_player=root_player,
        experiment_seed=task["experiment_seed"],
    )
    ply = 0
    while not game.over() and ply < 200:
        legal = game.possible_moves()
        if not legal:
            break
        search_seed = stable_seed(base_seed, stable_hash(game.to_state()), ply)
        search = PUCT(
            evaluator=_WORKER_EVALUATOR,
            simulations=int(task["continuation_budget"]),
            c_puct=SEARCH_C_PUCT,
            rng=random.Random(search_seed),
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
        )
        _, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
        move = int(search.select_root_move(root, legal))
        if not game.move(game.pit_index(move)):
            break
        ply += 1
    stores = game.captured_seeds
    winner = game.winner
    return {
        "kind": "deterministic",
        "state_hash": task["state_hash"],
        "continuation_budget": task["continuation_budget"],
        "forced_move": -1,
        "outcome_root": 0.0
        if winner is None
        else (1.0 if int(winner) == root_player else -1.0),
        "store_margin_root": int(stores[root_player] - stores[1 - root_player]),
        "paired_seed_context_hash": seed_context_hash,
    }


def _run_continuation_task(task: dict[str, Any]) -> dict[str, Any]:
    assert _WORKER_EVALUATOR is not None
    if task["kind"] == "forced":
        result = forced_continuation(
            evaluator=_WORKER_EVALUATOR, task=task, forced_move=task["forced_move"]
        )
        return {
            "kind": "forced",
            "state_hash": task["state_hash"],
            "continuation_budget": task["continuation_budget"],
            "forced_move": int(task["forced_move"]),
            "outcome_root": float(result["outcome_root"]),
            "store_margin_root": int(result["store_margin_root"]),
            "paired_seed_context_hash": result["paired_seed_context_hash"],
        }
    return _deterministic_continuation(task)


def continuation_experiment_seed(state_hash: str, budget: int) -> int:
    return stable_seed(CONTINUATION_NAMESPACE, state_hash, budget)


def read_cached_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def run_continuations(
    tasks: list[dict[str, Any]],
    *,
    checkpoint: Path,
    cache_path: Path,
    workers: int,
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    """Resume a unified cache of forced/deterministic continuations."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for record in read_cached_jsonl(cache_path):
        key = (
            str(record["kind"]),
            str(record["state_hash"]),
            int(record["continuation_budget"]),
            int(record.get("forced_move", -1)),
        )
        cache.setdefault(key, record)
    missing = []
    for task in tasks:
        key = (
            str(task["kind"]),
            str(task["state_hash"]),
            int(task["continuation_budget"]),
            int(task.get("forced_move", -1)),
        )
        if key not in cache:
            missing.append(task)
    if missing:
        worker_count = max(1, min(int(workers), len(missing)))
        with cache_path.open("a", encoding="utf-8") as stream:
            if worker_count == 1:
                _worker_initializer(str(checkpoint))
                for task in missing:
                    record = _run_continuation_task(task)
                    cache[
                        (
                            str(record["kind"]),
                            str(record["state_hash"]),
                            int(record["continuation_budget"]),
                            int(record.get("forced_move", -1)),
                        )
                    ] = record
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            else:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_worker_initializer,
                    initargs=(str(checkpoint),),
                ) as executor:
                    futures = [
                        executor.submit(_run_continuation_task, task)
                        for task in missing
                    ]
                    for future in as_completed(futures):
                        record = future.result()
                        cache[
                            (
                                str(record["kind"]),
                                str(record["state_hash"]),
                                int(record["continuation_budget"]),
                                int(record.get("forced_move", -1)),
                            )
                        ] = record
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                        stream.flush()
    return cache


def harmful_direction(
    diagnostic_batches: dict[str, list[dict[str, torch.Tensor]]],
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    device: torch.device,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Recompute PR #197's harmful grand-mean trunk gradient and Adam update."""
    model = new_model(device)
    model.load_state_dict(state)
    model.eval()
    lr = float(manifest["optimizer"]["lr"])
    clip = float(manifest["gradient_clip"])
    grads_by_shard = {
        shard: mean_full_gradient(model, diagnostic_batches[shard]) for shard in SHARDS
    }
    grand_grads = {
        name: torch.stack([grads_by_shard[shard][name] for shard in SHARDS]).mean(0)
        for name in grads_by_shard["S0"]
    }
    grand_update, _ = aggregate_update(
        state, optimizer_state, grand_grads, device, lr=lr, clip=clip
    )
    return {
        "harmful_trunk_gradient": group_flat(grand_grads, "shared_trunk"),
        "harmful_adam_trunk": grand_update["shared_trunk"],
        "per_shard_grads": grads_by_shard,
    }


def per_example_gradients(
    model: PolicyValueNet,
    train_rows: list[dict[str, Any]],
    probe: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Separate policy/value shared-trunk gradients for each probe state."""
    result: dict[str, dict[str, list[torch.Tensor]]] = {
        signal: {group: [] for group in GROUPS}
        for signal in ("policy", "value", "joint")
    }
    for rec in probe:
        batch = _batch(
            train_rows, np.asarray([rec["training_position"]], dtype=np.int64), device
        )
        vector = vectors(model, batch)
        for group in GROUPS:
            for signal in ("policy", "value", "joint"):
                result[signal][group].append(vector[group][signal])
    return {
        signal: {group: torch.stack(result[signal][group]) for group in GROUPS}
        for signal in result
    }


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the prespecified Phase I classification rules."""
    phase_c = summary.get("phase_c_policy_top_move", {})
    primary = phase_c.get("forced_continuation_primary", {})
    disagree_n = primary.get("disagreement_states", 0)
    by_budget = primary.get("by_budget", {})
    budgets = list(by_budget.keys())
    negative_at_both = all(by_budget[b]["margin_delta"]["mean"] < 0 for b in budgets)
    ci_zero_at_least_one = any(
        by_budget[b]["margin_delta"]["upper_95"] <= 0 for b in budgets
    )
    quartiles = primary.get("margin_delta_by_policy_residual_quartile", {})
    highest_q = quartiles.get("3", {}).get("mean", 0.0)
    other_q = np.mean([quartiles.get(q, {}).get("mean", 0.0) for q in ("0", "1", "2")])
    stronger_in_highest = highest_q < other_q
    misalign_grad = summary.get("phase_g_gradient_attribution", {}).get(
        "policy_misaligned_aggregate", {}
    )
    misalign_aligns_harm = float(misalign_grad.get("cosine_with_harmful", -1.0)) > 0

    policy_misaligned = (
        disagree_n >= 128
        and negative_at_both
        and ci_zero_at_least_one
        and stronger_in_highest
        and misalign_aligns_harm
    )

    phase_e = summary.get("phase_e_value_confirmation", {})
    d1200 = phase_e.get("sign_agreement_d1200", {})
    high_resid_agreement = d1200.get("high_residual_agreement", 1.0)
    phase_f = summary.get("phase_f_off_policy_diagnostic", {})
    match_agree = (
        phase_f.get("split", {})
        .get("replay_matches", {})
        .get("sign_agreement_d1200", 1.0)
    )
    differ_agree = (
        phase_f.get("split", {})
        .get("replay_differs", {})
        .get("sign_agreement_d1200", 1.0)
    )
    differ_worse = differ_agree < match_agree
    value_grad = summary.get("phase_g_gradient_attribution", {}).get(
        "value_disagrees_aggregate", {}
    )
    value_aligns_harm = float(value_grad.get("cosine_with_harmful", -1.0)) > 0

    value_problematic = (
        high_resid_agreement < 0.60 and differ_worse and value_aligns_harm
    )

    label = None
    next_action = None
    if policy_misaligned and value_problematic:
        policy_contrib = abs(
            float(
                summary.get("phase_g_gradient_attribution", {})
                .get("policy_misaligned_aggregate", {})
                .get("cosine_with_harmful", 0.0)
            )
        )
        value_contrib = abs(
            float(
                summary.get("phase_g_gradient_attribution", {})
                .get("value_disagrees_aggregate", {})
                .get("cosine_with_harmful", 0.0)
            )
        )
        label = "both_supervised_targets_problematic"
        first = "policy" if policy_contrib >= value_contrib else "value"
        next_action = (
            f"test one better-grounded {first} target first; do not change both "
            "channels in one training ablation"
        )
    elif policy_misaligned:
        label = "policy_targets_causally_misaligned"
        next_action = (
            "redesign policy targets; do not change value training simultaneously"
        )
    elif value_problematic:
        label = "value_targets_high_variance_or_off_policy"
        next_action = "test one better-grounded value target in a separate experiment"
    else:
        phase_g = summary.get("phase_g_gradient_attribution", {})
        both_contribute = (
            float(
                phase_g.get("policy_aligned_aggregate", {}).get(
                    "cosine_with_harmful", 0.0
                )
            )
            > 0
            and float(
                phase_g.get("value_confirmed_aggregate", {}).get(
                    "cosine_with_harmful", 0.0
                )
            )
            > 0
        )
        if both_contribute:
            label = "targets_individually_sound_objective_distillation_failure"
            next_action = (
                "investigate function-space/search-aware constraints on distillation "
                "rather than new targets"
            )
        else:
            label = "target_residual_quality_inconclusive"
            next_action = "sample sizes or CIs do not separate the hypotheses"

    return {
        "label": label,
        "next_action": next_action,
        "evidence": {
            "policy_disagreement_states": int(disagree_n),
            "policy_negative_at_both_budgets": bool(negative_at_both),
            "policy_ci_zero_at_least_one_budget": bool(ci_zero_at_least_one),
            "policy_misalignment_stronger_highest_quartile": bool(stronger_in_highest),
            "policy_misaligned_gradient_aligns_harmful": bool(misalign_aligns_harm),
            "value_high_residual_agreement_d1200": float(high_resid_agreement),
            "value_disagreement_worse_off_policy": bool(differ_worse),
            "value_questionable_gradient_aligns_harmful": bool(value_aligns_harm),
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-Lite Supervised Target-Residual Audit",
        "",
        f"**Classification:** `{summary['phase_i_classification']['label']}`",
        "",
        "**Primary question:** are the states driving the harmful PR #197 supervised "
        "update asking the network to prefer causally better moves and well-grounded "
        "value outcomes?",
        "",
        "## PR197 statistical repair",
        "",
    ]
    repair = summary["phase_a_pr197_statistical_repair"]
    phase_f = repair["phase_f_bootstrap_corrected"]
    median_est = phase_f["median_estimator"]
    mean_est = phase_f["mean_estimator"]
    lines.append(
        f"- Whole-game between-shard Adam cosine, median estimator: "
        f"{median_est['median']:.4f} (95% CI [{median_est['lower_95']:.4f}, {median_est['upper_95']:.4f}])"
    )
    lines.append(
        f"- Whole-game between-shard Adam cosine, mean estimator (labeled mean): "
        f"{mean_est['mean']:.4f} (95% CI [{mean_est['lower_95']:.4f}, {mean_est['upper_95']:.4f}])"
    )
    lines.extend(
        [
            "",
            "Direct between-minus-within bootstrap (no marginal-CI inference):",
            "",
            "| Effective rows | Difference (mean) | 95% CI |",
            "| ---: | ---: | ---: |",
        ]
    )
    for k in EFFECTIVE_BATCH_SIZES:
        entry = repair["phase_d_direct_bootstrap"][str(k)]
        diff = entry["between_minus_within_mean"]
        lines.append(
            f"| {k * 512} | {diff['point']:+.4f} | [{diff['lower_95']:+.4f}, {diff['upper_95']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Frozen probe provenance",
            "",
        ]
    )
    probe_manifest = summary["phase_b_frozen_probe"]["probe_manifest"]
    lines.append(
        f"- {probe_manifest['state_count']} unique training states, stratified 16x64 "
        "(policy-residual quartile x value-residual quartile), sub-stratified by "
        "phase/player and spread across policy-target entropy."
    )
    lines.append(
        f"- policy-residual quartile boundaries: "
        f"{[round(v, 4) for v in probe_manifest['policy_residual_quartile_boundaries']]}"
    )
    lines.append(
        f"- value-residual quartile boundaries: "
        f"{[round(v, 4) for v in probe_manifest['value_residual_quartile_boundaries']]}"
    )
    lines.append(f"- stratum counts: `{probe_manifest['stratum_counts']}`")
    lines.extend(
        [
            "",
            "## Policy residual distribution",
            "",
        ]
    )
    dist = summary["phase_b_frozen_probe"]["residual_distribution"]
    for key, label in (
        ("policy_residual_kl", "policy residual KL"),
        ("value_residual_abs", "value residual |target - value|"),
    ):
        entry = dist[key]
        lines.append(
            f"- {label}: mean {entry['mean']:.4f}, median {entry['median']:.4f}, "
            f"p90 {entry['p90']:.4f}, max {entry['max']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Policy top-move causal quality",
            "",
        ]
    )
    phase_c = summary["phase_c_policy_top_move"]
    agreement = phase_c["top_move_agreement"]
    lines.append(
        f"- raw-policy vs replay top move: {agreement['raw_vs_replay']:.3f}; "
        f"raw-policy vs D384: {agreement['raw_vs_d384']:.3f}; "
        f"raw-policy vs D1200: {agreement['raw_vs_d1200']:.3f}; "
        f"replay vs D384: {agreement['replay_vs_d384']:.3f}; "
        f"replay vs D1200: {agreement['replay_vs_d1200']:.3f}"
    )
    primary = phase_c["forced_continuation_primary"]
    lines.append(
        f"- replay-target top move vs current D384 top move (forced, current/current "
        f"continuation), disagreement states: {primary['disagreement_states']}"
    )
    lines.extend(
        [
            "",
            "| Budget | Margin delta (replay - current) | 95% CI | Outcome delta |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for budget in PHASE_C_CONTINUATION_BUDGETS:
        entry = primary["by_budget"][str(budget)]
        margin = entry["margin_delta"]
        outcome = entry["outcome_delta"]
        lines.append(
            f"| {budget} | {margin['mean']:+.4f} | [{margin['lower_95']:+.4f}, {margin['upper_95']:+.4f}] "
            f"| {outcome['mean']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Margin delta by policy-residual quartile (1200 budget):",
            "",
            "| Quartile | Margin delta |",
            "| ---: | ---: |",
        ]
    )
    for q in ("0", "1", "2", "3"):
        entry = primary["margin_delta_by_policy_residual_quartile"].get(q, {})
        lines.append(f"| {q} | {entry.get('mean', 0.0):+.4f} |")
    lines.extend(
        [
            "",
            "## Distribution-level gradient/quality alignment",
            "",
        ]
    )
    phase_d = summary["phase_d_distribution_alignment"]
    lines.append(f"- bounded subset states: {phase_d['subset_states']}")
    alignment = phase_d["alignment"]
    concordance = phase_d["pairwise_concordance"]
    expected = phase_d["expected_causal_quality_change"]
    lines.append(
        f"- gradient-quality alignment (cosine of replay mass-shift vs centered quality): "
        f"mean {alignment['mean']:+.4f} (95% CI [{alignment['lower_95']:+.4f}, {alignment['upper_95']:+.4f}])"
    )
    lines.append(
        f"- pairwise concordance: {concordance['mean']:.4f} (95% CI [{concordance['lower_95']:.4f}, {concordance['upper_95']:.4f}])"
    )
    lines.append(
        f"- expected causal-quality change under infinitesimal policy step: "
        f"mean {expected['mean']:+.4f} (95% CI [{expected['lower_95']:+.4f}, {expected['upper_95']:+.4f}])"
    )
    lines.extend(
        [
            "",
            "## Value residual confirmation",
            "",
        ]
    )
    phase_e = summary["phase_e_value_confirmation"]
    d1200 = phase_e["sign_agreement_d1200"]
    d768 = phase_e["sign_agreement_d768"]
    sqerr = phase_e["squared_error_change"]
    lines.append(
        f"- residual sign agreement with D1200 outcome: {d1200['overall']:.3f} "
        f"(high-residual: {d1200['high_residual_agreement']:.3f})"
    )
    lines.append(f"- residual sign agreement with D768 outcome: {d768['overall']:.3f}")
    lines.append(
        f"- squared-error change if following replay target (vs D1200 reference): "
        f"mean {sqerr['mean']:+.4f}"
    )
    lines.extend(
        [
            "",
            "## Exploratory-vs-deterministic value-target analysis",
            "",
        ]
    )
    phase_f = summary["phase_f_off_policy_diagnostic"]
    lines.append(
        f"- replay immediate move matches current D384 search: {phase_f['match_rate']:.3f}"
    )
    for split_key, split_label in (
        ("replay_matches", "replay move matches D384"),
        ("replay_differs", "replay move differs from D384"),
    ):
        split = phase_f["split"].get(split_key, {})
        lines.append(
            f"- {split_label}: sign agreement {split.get('sign_agreement_d1200', 0.0):.3f} "
            f"(n={split.get('informative_n', 0)})"
        )
    for temp_key, temp_label in (
        ("early_high_temp", "early high-temp region"),
        ("later_low_temp", "later low-temp region"),
    ):
        split = phase_f["temperature_split"].get(temp_key, {})
        lines.append(
            f"- {temp_label}: sign agreement {split.get('sign_agreement_d1200', 0.0):.3f} "
            f"(n={split.get('informative_n', 0)})"
        )
    lines.extend(
        [
            "",
            "## Harmful-gradient attribution",
            "",
        ]
    )
    phase_g = summary["phase_g_gradient_attribution"]
    for key, label in (
        ("policy_aligned_aggregate", "policy aligned rows"),
        ("policy_misaligned_aggregate", "policy misaligned rows"),
        ("value_confirmed_aggregate", "value confirmed rows"),
        ("value_disagrees_aggregate", "value disagrees rows"),
    ):
        entry = phase_g.get(key, {})
        lines.append(
            f"- {label} (n={entry.get('n', 0)}): cosine with harmful trunk gradient "
            f"{entry.get('cosine_with_harmful', 0.0):+.4f}"
        )
    lines.extend(
        [
            "",
            "## Counterfactual objective-quality estimate",
            "",
        ]
    )
    phase_h = summary["phase_h_counterfactual"]
    for key, label in (
        ("G_all", "G_all (exact normal objective)"),
        ("G_policy_aligned_only", "G_policy_aligned_only"),
        ("G_value_confirmed_only", "G_value_confirmed_only"),
        ("G_both_filtered", "G_both_filtered"),
    ):
        entry = phase_h[key]
        lines.append(
            f"- {label}: cosine with harmful {entry['cosine_with_harmful']:+.4f}, "
            f"norm {entry['raw_norm']:.4f}, Adam-step cosine {entry['adam_cosine']:+.4f}"
        )
    lines.extend(
        [
            "",
            "## Exact classification evidence",
            "",
            "| Signal | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in summary["phase_i_classification"]["evidence"].items():
        if isinstance(value, bool):
            rendered = str(value)
        elif isinstance(value, int):
            rendered = str(value)
        else:
            rendered = f"{value:.4f}"
        lines.append(f"| {key} | {rendered} |")
    lines.extend(
        [
            "",
            "## One next action",
            "",
            f"`{summary['phase_i_classification']['next_action']}`",
            "",
            "Full evidence: `docs/data/alphazero-lite-supervised-target-residual-audit-summary.json`.",
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
        default=Path("/tmp/azlite_supervised_target_residual_audit"),
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-supervised-target-residual-audit-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-supervised-target-residual-audit-results.md",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--phase-d-subset", type=int, default=PHASE_D_SUBSET)
    args = parser.parse_args()

    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact does not match PR191")
    device = torch.device("cpu")
    configure_determinism(device, int(manifest["seed"]))

    all_rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(
        manifest["artifact_paths"]["train_source_indexes"], allow_pickle=False
    )
    rows = [all_rows[int(index)] for index in source]

    state, optimizer_state = fresh_state(manifest, device)
    model = new_model(device)
    model.load_state_dict(state)
    model.eval()

    x_all = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask_all = legal_mask_matrix_for_encoded_states(x_all)
    current_policy, current_value = current_policy_value(model, x_all, mask_all, device)

    # Phase B — frozen target-residual probe.
    probe, probe_manifest = build_probe(
        rows, source, current_policy, current_value, seed=int(manifest["seed"])
    )
    probe_manifest["manifest_sha256"] = stable_hash(probe_manifest)
    probe_manifest["replay_sha256"] = sha256_file(Path(manifest["replay_path"]))
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "probe.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in probe),
        encoding="utf-8",
    )
    (args.workdir / "probe_manifest.json").write_text(
        json.dumps(probe_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    residuals = {
        "policy_residual_kl": {
            "mean": float(np.mean([row["policy_residual_kl"] for row in probe])),
            "median": float(np.median([row["policy_residual_kl"] for row in probe])),
            "p90": float(
                np.quantile([row["policy_residual_kl"] for row in probe], 0.9)
            ),
            "max": float(max(row["policy_residual_kl"] for row in probe)),
        },
        "value_residual_abs": {
            "mean": float(np.mean([row["value_residual_abs"] for row in probe])),
            "median": float(np.median([row["value_residual_abs"] for row in probe])),
            "p90": float(
                np.quantile([row["value_residual_abs"] for row in probe], 0.9)
            ),
            "max": float(max(row["value_residual_abs"] for row in probe)),
        },
    }

    # Phase A — PR #197 statistical repair (gradients only; no arena).
    bootstrap_cache = Path(
        "/tmp/azlite_aggregate_gradient_stability/bootstrap_medians.json"
    )
    phase_f_corrected: dict[str, Any] = {"replicates": 0}
    if bootstrap_cache.is_file():
        medians = json.loads(bootstrap_cache.read_text(encoding="utf-8"))
        phase_f_corrected = {
            "replicates": len(medians),
            "median_estimator": bootstrap_median_ci(medians, seed=304),
            "mean_estimator": bootstrap_mean_ci(medians, seed=304),
        }
    assignments, shard_manifest = partition(rows)
    diagnostic_batches = {
        name: [
            _batch(rows, indexes, device)
            for indexes in deterministic_batches(indexes, name)
        ]
        for name, indexes in assignments.items()
    }
    trunk_updates: dict[str, dict[int, list[torch.Tensor]]] = {
        shard: {k: [] for k in EFFECTIVE_BATCH_SIZES} for shard in SHARDS
    }
    for shard in SHARDS:
        indexes = assignments[shard]
        for k in EFFECTIVE_BATCH_SIZES:
            for replicate in range(AGGREGATE_REPLICATES):
                micro = aggregate_microbatches(indexes, shard, k, replicate)
                batches = [_batch(rows, idx, device) for idx in micro]
                grads = mean_full_gradient(model, batches)
                update, _ = aggregate_update(
                    state,
                    optimizer_state,
                    grads,
                    device,
                    lr=float(manifest["optimizer"]["lr"]),
                    clip=float(manifest["gradient_clip"]),
                )
                trunk_updates[shard][k].append(update["shared_trunk"])
    phase_d_direct: dict[str, Any] = {}
    for k in EFFECTIVE_BATCH_SIZES:
        within: list[float] = []
        between: list[float] = []
        for shard in SHARDS:
            reps = trunk_updates[shard][k]
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    within.append(cosine(reps[i], reps[j]))
        for r in range(AGGREGATE_REPLICATES):
            for i in range(len(SHARDS)):
                for j in range(i + 1, len(SHARDS)):
                    between.append(
                        cosine(
                            trunk_updates[SHARDS[i]][k][r],
                            trunk_updates[SHARDS[j]][k][r],
                        )
                    )
        phase_d_direct[str(k)] = {
            "within_shard_mean": bootstrap_mean_ci(within, seed=301),
            "between_shard_mean": bootstrap_mean_ci(between, seed=302),
            "between_minus_within_mean": bootstrap_difference_ci(
                between, within, seed=303, statistic="mean"
            ),
            "between_minus_within_median": bootstrap_difference_ci(
                between, within, seed=303, statistic="median"
            ),
        }
    phase_a = {
        "phase_f_bootstrap_corrected": phase_f_corrected,
        "phase_d_direct_bootstrap": phase_d_direct,
        "shard_hashes": {
            shard: shard_manifest["shards"][shard]["shard_sha256"] for shard in SHARDS
        },
    }

    # Harmful grand-mean direction (raw gradient + Adam update).
    harmful = harmful_direction(
        diagnostic_batches, state, optimizer_state, device, manifest
    )
    harmful_gradient = harmful["harmful_trunk_gradient"]
    harmful_adam = harmful["harmful_adam_trunk"]

    # Phase C — policy top-move quality (search + forced continuations).
    evaluator = arena.ArtifactEvaluator(args.current)
    top_move_cache_path = args.workdir / "top_move_records.jsonl"
    top_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for record in read_cached_jsonl(top_move_cache_path):
        top_cache[(str(record["state_hash"]), int(record["simulations"]))] = record
    top_missing = []
    for rec in probe:
        seed, _ = search_seed_for_state(rec, probe_manifest["manifest_sha256"])
        for sims in PHASE_C_SEARCH_BUDGETS:
            if (rec["state_hash"], sims) not in top_cache:
                top_missing.append((rec, sims, seed))
    if top_missing:
        top_move_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with top_move_cache_path.open("a", encoding="utf-8") as stream:
            for rec, sims, seed in top_missing:
                record = run_top_move_search(rec, evaluator, sims, seed)
                top_cache[(rec["state_hash"], sims)] = record
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    raw_vs_replay = []
    raw_vs_d384 = []
    raw_vs_d1200 = []
    replay_vs_d384 = []
    replay_vs_d1200 = []
    continuation_tasks: dict[tuple[str, str, int, int], dict[str, Any]] = {}

    for rec in probe:
        raw_top = rec["current_raw_top_move"]
        replay_top = rec["replay_top_move"]
        d384 = top_cache[(rec["state_hash"], 384)]["selected_move"]
        d1200 = top_cache[(rec["state_hash"], 1200)]["selected_move"]
        raw_vs_replay.append(int(raw_top == replay_top))
        raw_vs_d384.append(int(raw_top == d384))
        raw_vs_d1200.append(int(raw_top == d1200))
        replay_vs_d384.append(int(replay_top == d384))
        replay_vs_d1200.append(int(replay_top == d1200))
        for budget in PHASE_C_CONTINUATION_BUDGETS:
            seed = continuation_experiment_seed(rec["state_hash"], budget)
            if replay_top != d384:
                for move in (replay_top, d384):
                    key = ("forced", rec["state_hash"], budget, move)
                    continuation_tasks[key] = {
                        "kind": "forced",
                        "state": rec["state"],
                        "state_hash": rec["state_hash"],
                        "continuation_budget": budget,
                        "experiment_seed": seed,
                        "forced_move": move,
                    }
            if replay_top != d1200:
                for move in (replay_top, d1200):
                    key = ("forced", rec["state_hash"], budget, move)
                    continuation_tasks[key] = {
                        "kind": "forced",
                        "state": rec["state"],
                        "state_hash": rec["state_hash"],
                        "continuation_budget": budget,
                        "experiment_seed": seed,
                        "forced_move": move,
                    }

    # Phase E deterministic reference continuations.
    for rec in probe:
        for budget in PHASE_E_CONTINUATION_BUDGETS:
            key = ("deterministic", rec["state_hash"], budget, -1)
            continuation_tasks[key] = {
                "kind": "deterministic",
                "state": rec["state"],
                "state_hash": rec["state_hash"],
                "continuation_budget": budget,
                "experiment_seed": continuation_experiment_seed(
                    rec["state_hash"], budget
                ),
            }

    # Phase D forced move-quality continuations (bounded subset).
    phase_d_indexes = list(
        range(0, len(probe), max(1, len(probe) // args.phase_d_subset))
    )[: args.phase_d_subset]
    phase_d_rows = [probe[i] for i in phase_d_indexes]
    phase_d_selected_moves: dict[str, list[int]] = {}
    for rec in phase_d_rows:
        legal = rec["legal_moves"]
        if len(legal) <= 4:
            selected = list(legal)
        else:
            raw_top2 = sorted(legal, key=lambda m: (-rec["current_policy"][m], m))[:2]
            replay_top2 = sorted(legal, key=lambda m: (-rec["replay_policy"][m], m))[:2]
            d384_visits = {
                item["move"]: item["visits"]
                for item in top_cache[(rec["state_hash"], 384)]["child_stats"]
            }
            d1200_visits = {
                item["move"]: item["visits"]
                for item in top_cache[(rec["state_hash"], 1200)]["child_stats"]
            }
            d384_top2 = sorted(legal, key=lambda m: (-d384_visits.get(m, 0), m))[:2]
            d1200_top2 = sorted(legal, key=lambda m: (-d1200_visits.get(m, 0), m))[:2]
            seen: set[int] = set()
            selected = []
            for move in [*raw_top2, *replay_top2, *d384_top2, *d1200_top2]:
                if move not in seen:
                    seen.add(move)
                    selected.append(move)
        phase_d_selected_moves[rec["state_hash"]] = [int(move) for move in selected]
        for move in selected:
            key = ("forced", rec["state_hash"], PHASE_D_CONTINUATION_BUDGET, int(move))
            continuation_tasks[key] = {
                "kind": "forced",
                "state": rec["state"],
                "state_hash": rec["state_hash"],
                "continuation_budget": PHASE_D_CONTINUATION_BUDGET,
                "experiment_seed": continuation_experiment_seed(
                    rec["state_hash"], PHASE_D_CONTINUATION_BUDGET
                ),
                "forced_move": int(move),
            }

    continuation_cache = run_continuations(
        list(continuation_tasks.values()),
        checkpoint=Path(manifest["artifact_paths"]["initialization_checkpoint"]),
        cache_path=args.workdir / "continuation_records.jsonl",
        workers=args.workers,
    )

    def continuation(state_hash: str, budget: int, move: int | None) -> dict[str, Any]:
        key = (
            "forced" if move is not None else "deterministic",
            state_hash,
            budget,
            int(move if move is not None else -1),
        )
        return continuation_cache[key]

    # Phase C forced-continuation comparisons.
    def paired_deltas(
        rec: dict[str, Any], move_a: int, move_b: int, budget: int
    ) -> tuple[float, float]:
        a = continuation(rec["state_hash"], budget, move_a)
        b = continuation(rec["state_hash"], budget, move_b)
        margin = (a["store_margin_root"] - b["store_margin_root"]) / 48.0
        outcome = a["outcome_root"] - b["outcome_root"]
        return margin, outcome

    primary_margins: dict[int, list[float]] = defaultdict(list)
    primary_outcomes: dict[int, list[float]] = defaultdict(list)
    primary_by_quartile: dict[str, dict[str, float]] = defaultdict(dict)
    primary_states: set[str] = set()
    replay_vs_d1200_margins: dict[int, list[float]] = defaultdict(list)
    replay_vs_d1200_outcomes: dict[int, list[float]] = defaultdict(list)
    for rec in probe:
        d384 = top_cache[(rec["state_hash"], 384)]["selected_move"]
        d1200 = top_cache[(rec["state_hash"], 1200)]["selected_move"]
        replay_top = rec["replay_top_move"]
        if replay_top != d384:
            primary_states.add(rec["state_hash"])
            for budget in PHASE_C_CONTINUATION_BUDGETS:
                margin, outcome = paired_deltas(rec, replay_top, d384, budget)
                primary_margins[budget].append(margin)
                primary_outcomes[budget].append(outcome)
                if budget == 1200:
                    primary_by_quartile[str(rec["policy_residual_quartile"])][
                        rec["state_hash"]
                    ] = margin
        if replay_top != d1200:
            for budget in PHASE_C_CONTINUATION_BUDGETS:
                margin, outcome = paired_deltas(rec, replay_top, d1200, budget)
                replay_vs_d1200_margins[budget].append(margin)
                replay_vs_d1200_outcomes[budget].append(outcome)

    primary_by_budget: dict[str, Any] = {}
    for budget in PHASE_C_CONTINUATION_BUDGETS:
        primary_by_budget[str(budget)] = {
            "margin_delta": bootstrap_paired_ci(primary_margins[budget], seed=401),
            "outcome_delta": bootstrap_paired_ci(primary_outcomes[budget], seed=402),
        }
    margin_by_quartile: dict[str, Any] = {}
    for q in ("0", "1", "2", "3"):
        values = list(primary_by_quartile[q].values())
        margin_by_quartile[q] = bootstrap_paired_ci(values, seed=403 + int(q))

    replay_vs_d1200_by_budget: dict[str, Any] = {}
    for budget in PHASE_C_CONTINUATION_BUDGETS:
        replay_vs_d1200_by_budget[str(budget)] = {
            "margin_delta": bootstrap_paired_ci(
                replay_vs_d1200_margins[budget], seed=404
            ),
            "outcome_delta": bootstrap_paired_ci(
                replay_vs_d1200_outcomes[budget], seed=405
            ),
        }

    top_move_agreement = {
        "raw_vs_replay": float(np.mean(raw_vs_replay)),
        "raw_vs_d384": float(np.mean(raw_vs_d384)),
        "raw_vs_d1200": float(np.mean(raw_vs_d1200)),
        "replay_vs_d384": float(np.mean(replay_vs_d384)),
        "replay_vs_d1200": float(np.mean(replay_vs_d1200)),
    }

    # Phase D — distribution-level alignment.
    alignment: list[float] = []
    concordance: list[float] = []
    expected_change: list[float] = []
    phase_d_by_quartile: dict[str, list[float]] = defaultdict(list)
    for rec in phase_d_rows:
        legal = rec["legal_moves"]
        selected = phase_d_selected_moves[rec["state_hash"]]
        current = np.asarray(rec["current_policy"], dtype=np.float64)
        target = np.asarray(rec["replay_policy"], dtype=np.float64)
        mass_shift = target - current  # gradient-descent push in logit space
        margins = []
        for move in selected:
            record = continuation(rec["state_hash"], PHASE_D_CONTINUATION_BUDGET, move)
            margins.append(record["store_margin_root"] / 48.0)
        margins = np.asarray(margins, dtype=np.float64)
        centered = margins - margins.mean()
        d = mass_shift[selected]
        if np.linalg.norm(centered) > 1e-12 and np.linalg.norm(d) > 1e-12:
            alignment.append(
                float(
                    np.dot(d, centered)
                    / (np.linalg.norm(d) * np.linalg.norm(centered) + 1e-20)
                )
            )
        else:
            alignment.append(0.0)
        pairs = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                pairs.append(
                    int(np.sign(d[i] - d[j]) == np.sign(centered[i] - centered[j]))
                )
        concordance.append(float(np.mean(pairs)) if pairs else 0.0)
        expected_change.append(float(np.dot(d, centered)))
        phase_d_by_quartile[str(rec["policy_residual_quartile"])].append(
            float(np.dot(d, centered))
        )

    phase_d_summary = {
        "subset_states": len(phase_d_rows),
        "continuation_budget": PHASE_D_CONTINUATION_BUDGET,
        "alignment": bootstrap_paired_ci(alignment, seed=501),
        "pairwise_concordance": bootstrap_paired_ci(concordance, seed=502),
        "expected_causal_quality_change": bootstrap_paired_ci(
            expected_change, seed=503
        ),
        "expected_change_by_policy_residual_quartile": {
            q: bootstrap_paired_ci(values, seed=504 + int(q))
            for q, values in phase_d_by_quartile.items()
        },
    }

    # Phase E — value residual confirmation.
    def sign_agreement(rows_subset: list[dict[str, Any]], budget: int) -> list[int]:
        informative = []
        for rec in rows_subset:
            residual_dir = np.sign(
                float(rec["replay_value"]) - float(rec["current_value"])
            )
            if residual_dir == 0:
                continue
            ref = continuation(rec["state_hash"], budget, None)
            ref_dir = np.sign(ref["outcome_root"] - float(rec["current_value"]))
            informative.append(int(ref_dir == residual_dir))
        return informative

    d1200_info = sign_agreement(probe, 1200)
    d768_info = sign_agreement(probe, 768)
    high_resid = [
        rec
        for rec in probe
        if rec["policy_residual_quartile"] == 3 or rec["value_residual_quartile"] == 3
    ]
    high_resid_info = sign_agreement(high_resid, 1200)

    def squared_error_change(
        rows_subset: list[dict[str, Any]], budget: int
    ) -> list[float]:
        out = []
        for rec in rows_subset:
            ref = continuation(rec["state_hash"], budget, None)
            current_err = (float(rec["current_value"]) - ref["outcome_root"]) ** 2
            follow_err = (float(rec["replay_value"]) - ref["outcome_root"]) ** 2
            out.append(follow_err - current_err)
        return out

    sqerr_change = squared_error_change(probe, 1200)
    phase_e_summary = {
        "sign_agreement_d1200": {
            "overall": float(np.mean(d1200_info)),
            "informative_n": len(d1200_info),
            "high_residual_agreement": float(np.mean(high_resid_info)),
            "by_residual_quartile": {
                q: float(
                    np.mean(
                        sign_agreement(
                            [
                                rec
                                for rec in probe
                                if rec["value_residual_quartile"] == int(q)
                            ],
                            1200,
                        )
                        or [0.0]
                    )
                )
                for q in ("0", "1", "2", "3")
            },
            "by_phase": {
                ph: float(
                    np.mean(
                        sign_agreement(
                            [rec for rec in probe if rec["phase"] == ph], 1200
                        )
                        or [0.0]
                    )
                )
                for ph in ("opening", "mid", "late")
            },
            "by_player": {
                str(pl): float(
                    np.mean(
                        sign_agreement(
                            [rec for rec in probe if rec["player"] == pl], 1200
                        )
                        or [0.0]
                    )
                )
                for pl in (0, 1)
            },
        },
        "sign_agreement_d768": {
            "overall": float(np.mean(d768_info)),
            "informative_n": len(d768_info),
        },
        "squared_error_change": {
            "mean": float(np.mean(sqerr_change)),
            "median": float(np.median(sqerr_change)),
        },
    }

    # Phase F — off-policy diagnostic.
    def replay_played_move(rec: dict[str, Any]) -> int | None:
        source_index = rec["source_index"]
        if source_index + 1 >= len(all_rows):
            return None
        nxt = all_rows[source_index + 1]
        if (
            nxt.get("trajectory_hash") != rec["trajectory_hash"]
            or nxt.get("game_index") != rec["game_index"]
        ):
            return None
        current_state = rec["state"]
        next_state = _decoded_state(nxt)
        game = KalahGame.from_state(current_state)
        for move in game.possible_moves():
            clone = game.clone()
            if not clone.move(clone.pit_index(move)):
                continue
            if clone.to_state() == next_state:
                return int(move)
        return None

    played_moves: dict[str, int | None] = {}
    for rec in probe:
        played_moves[rec["state_hash"]] = replay_played_move(rec)

    matches = []
    differs = []
    for rec in probe:
        d384 = top_cache[(rec["state_hash"], 384)]["selected_move"]
        played = played_moves[rec["state_hash"]]
        if played is None:
            continue
        if played == d384:
            matches.append(rec)
        else:
            differs.append(rec)
    match_info = sign_agreement(matches, 1200)
    differ_info = sign_agreement(differs, 1200)

    early_rows = [rec for rec in probe if rec["action_sampling_noise_enabled"]]
    late_rows = [rec for rec in probe if not rec["action_sampling_noise_enabled"]]
    early_info = sign_agreement(early_rows, 1200)
    late_info = sign_agreement(late_rows, 1200)

    phase_f_summary = {
        "match_rate": float(
            np.mean(
                [
                    int(
                        played_moves[rec["state_hash"]]
                        == top_cache[(rec["state_hash"], 384)]["selected_move"]
                    )
                    for rec in probe
                    if played_moves[rec["state_hash"]] is not None
                ]
            )
        ),
        "split": {
            "replay_matches": {
                "informative_n": len(match_info),
                "sign_agreement_d1200": float(np.mean(match_info))
                if match_info
                else 0.0,
            },
            "replay_differs": {
                "informative_n": len(differ_info),
                "sign_agreement_d1200": float(np.mean(differ_info))
                if differ_info
                else 0.0,
            },
        },
        "temperature_split": {
            "early_high_temp": {
                "informative_n": len(early_info),
                "sign_agreement_d1200": float(np.mean(early_info))
                if early_info
                else 0.0,
            },
            "later_low_temp": {
                "informative_n": len(late_info),
                "sign_agreement_d1200": float(np.mean(late_info)) if late_info else 0.0,
            },
        },
    }

    # Phase G — per-example gradient attribution.
    per_example = per_example_gradients(model, rows, probe, device)
    policy_grads = per_example["policy"]["shared_trunk"]  # (N, D)
    value_grads = per_example["value"]["shared_trunk"]

    policy_misaligned_mask = np.zeros(len(probe), dtype=bool)
    for idx, rec in enumerate(probe):
        d384 = top_cache[(rec["state_hash"], 384)]["selected_move"]
        if rec["replay_top_move"] == d384:
            continue
        margins = []
        for budget in PHASE_C_CONTINUATION_BUDGETS:
            a = continuation(rec["state_hash"], budget, rec["replay_top_move"])
            b = continuation(rec["state_hash"], budget, d384)
            margins.append((a["store_margin_root"] - b["store_margin_root"]) / 48.0)
        policy_misaligned_mask[idx] = float(np.mean(margins)) < 0

    value_confirmed_mask = np.ones(len(probe), dtype=bool)
    for idx, rec in enumerate(probe):
        residual_dir = np.sign(float(rec["replay_value"]) - float(rec["current_value"]))
        if residual_dir == 0:
            continue
        ref = continuation(rec["state_hash"], 1200, None)
        ref_dir = np.sign(ref["outcome_root"] - float(rec["current_value"]))
        value_confirmed_mask[idx] = ref_dir == residual_dir

    def aggregate_cosine(grad_tensor: torch.Tensor) -> dict[str, Any]:
        if grad_tensor.shape[0] == 0:
            return {"cosine_with_harmful": 0.0, "raw_norm": 0.0}
        agg = grad_tensor.mean(0)
        return {
            "cosine_with_harmful": cosine(agg, harmful_gradient),
            "raw_norm": float(torch.linalg.vector_norm(agg)),
        }

    phase_g_summary = {
        "harmful_trunk_gradient_sha256": hashlib.sha256(
            harmful_gradient.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "policy_aligned_aggregate": {
            "n": int((~policy_misaligned_mask).sum()),
            **aggregate_cosine(policy_grads[~policy_misaligned_mask]),
        },
        "policy_misaligned_aggregate": {
            "n": int(policy_misaligned_mask.sum()),
            **aggregate_cosine(policy_grads[policy_misaligned_mask]),
        },
        "value_confirmed_aggregate": {
            "n": int(value_confirmed_mask.sum()),
            **aggregate_cosine(value_grads[value_confirmed_mask]),
        },
        "value_disagrees_aggregate": {
            "n": int((~value_confirmed_mask).sum()),
            **aggregate_cosine(value_grads[~value_confirmed_mask]),
        },
    }

    # Phase H — counterfactual objective-quality estimate.
    g_all = policy_grads.mean(0) + value_grads.mean(0)
    g_policy_aligned = policy_grads[~policy_misaligned_mask].mean(0) + value_grads.mean(
        0
    )
    g_value_confirmed = policy_grads.mean(0) + value_grads[value_confirmed_mask].mean(0)
    g_both = policy_grads[~policy_misaligned_mask].mean(0) + value_grads[
        value_confirmed_mask
    ].mean(0)

    def virtual_gradient_report(grad: torch.Tensor) -> dict[str, Any]:
        full_grads: dict[str, torch.Tensor] = {}
        offset = 0
        for pname, param in model.named_parameters():
            if parameter_group(pname) != "shared_trunk":
                full_grads[pname] = torch.zeros_like(param)
                continue
            count = int(param.numel())
            full_grads[pname] = (
                grad[offset : offset + count].reshape(param.shape).to(device)
            )
            offset += count
        update, _ = aggregate_update(
            state,
            optimizer_state,
            full_grads,
            device,
            lr=float(manifest["optimizer"]["lr"]),
            clip=float(manifest["gradient_clip"]),
        )
        adam_trunk = update["shared_trunk"]
        return {
            "cosine_with_harmful": cosine(grad, harmful_gradient),
            "raw_norm": float(torch.linalg.vector_norm(grad)),
            "adam_cosine": cosine(adam_trunk, harmful_adam),
        }

    phase_h_summary = {
        "G_all": virtual_gradient_report(g_all),
        "G_policy_aligned_only": virtual_gradient_report(g_policy_aligned),
        "G_value_confirmed_only": virtual_gradient_report(g_value_confirmed),
        "G_both_filtered": virtual_gradient_report(g_both),
    }

    summary: dict[str, Any] = {
        "schema": "azlite_supervised_target_residual_audit_v1",
        "guardrails": {
            "training": False,
            "optimizer_steps_that_mutate_candidates": False,
            "new_self_play": False,
            "replay_expansion": False,
            "stronger_teacher": False,
            "denoising": False,
            "margin_value_target": False,
            "lr_or_loss_weight_tuning": False,
            "behavior_anchor": False,
            "c_puct_tuning": False,
            "promotion": False,
        },
        "inputs": {
            "current_weights_sha256": CURRENT_HASH,
            "replay_sha256": sha256_file(Path(manifest["replay_path"])),
            "train_rows": len(rows),
            "probe_size": PROBE_SIZE,
        },
        "phase_a_pr197_statistical_repair": phase_a,
        "phase_b_frozen_probe": {
            "probe_manifest": probe_manifest,
            "residual_distribution": residuals,
        },
        "phase_c_policy_top_move": {
            "top_move_agreement": top_move_agreement,
            "forced_continuation_primary": {
                "disagreement_states": len(primary_states),
                "by_budget": primary_by_budget,
                "margin_delta_by_policy_residual_quartile": margin_by_quartile,
            },
            "forced_continuation_replay_vs_d1200": {
                "by_budget": replay_vs_d1200_by_budget,
            },
            "replay_targets_persisted": True,
        },
        "phase_d_distribution_alignment": phase_d_summary,
        "phase_e_value_confirmation": phase_e_summary,
        "phase_f_off_policy_diagnostic": phase_f_summary,
        "phase_g_gradient_attribution": phase_g_summary,
        "phase_h_counterfactual": phase_h_summary,
    }
    summary["phase_i_classification"] = classify(summary)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary["phase_i_classification"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
