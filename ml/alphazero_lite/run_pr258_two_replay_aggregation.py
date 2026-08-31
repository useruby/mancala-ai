#!/usr/bin/env python3
# ruff: noqa: E402
"""Sealed two-independent-replay aggregation test at fixed SGD compute."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import consumed_suite_registry as registry_module
from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as replay
from ml.alphazero_lite import run_pr241_optimizer_isolation_reproduction as contract
from ml.alphazero_lite import run_pr241_policy_target_noise_isolation as isolation
from ml.alphazero_lite import run_pr242_target_entropy_factorization as pr244
from ml.alphazero_lite import run_pr249_fresh_suite_generalization as pr249
from ml.alphazero_lite import run_pr251_cross_seed_strength_residual_transfer as pr251
from ml.alphazero_lite.evaluation_metrics import (
    paired_effect_difference,
    paired_opening_candidate_effect,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    game_split,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    ADAPTER_KEYS,
    export,
    new_model,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (
    _cross_entropy,
    incumbent_policy_batch,
    mixed_policy_target,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch, _losses
from ml.alphazero_lite.run_shared_trunk_delta_attribution import js
from ml.alphazero_lite.self_play import (
    CheckpointEvaluator,
    Node,
    PUCT,
    build_policy_target,
    build_search_profile,
    derive_self_play_value_target,
    encode_state,
    outcome_for_player,
    sample_move,
    standard_start_state,
    trajectory_hash_for_encoded_states,
)
from ml.alphazero_lite.train import (
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

PAIRS = ((53, 54), (55, 56), (57, 58), (59, 60), (61, 62))
SEEDS = tuple(seed for pair in PAIRS for seed in pair)
SUITE_SEEDS = {"V": 22042, "W": 23042, "X": 24042}
GAMES, WORKERS, SIMULATIONS, MAX_MOVES = 700, 24, 384, 200
STEPS, BATCH_SIZE, EXAMPLES = (1, 4, 16), 512, 8192
BETA, LR, PLAN_SEED = 0.95, 1e-5, 258042
TARGET = Path("/tmp/azlite_fresh_p1_parent_adapter/artifacts/step_0016/checkpoint.npz")
A16_SHA = "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
ADAM_SHA = "61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7"
P1_SHA = "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9"
TARGET_SHA = "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34"
_EVALUATOR: CheckpointEvaluator | None = None


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def worker_for(index: int) -> int:
    cursor = 0
    for worker, count in enumerate([30] * 4 + [29] * 20):
        if cursor <= index < cursor + count:
            return worker
        cursor += count
    fail("game index outside contract")


def game_rng(seed: int, index: int) -> random.Random:
    return random.Random(seed * 1_000_003 + index + worker_for(index) * 9_973)


def init_worker(checkpoint: str) -> None:
    global _EVALUATOR
    _EVALUATOR = CheckpointEvaluator(Path(checkpoint), input_encoding="kalah_v3")


def generate_game(item: tuple[int, int]) -> tuple[int, list[dict[str, Any]]]:
    seed, index = item
    game, rng, root, records = (
        KalahGame.from_state(standard_start_state()),
        game_rng(seed, index),
        None,
        [],
    )
    for ply in range(MAX_MOVES):
        if game.over():
            break
        legal = game.possible_moves()
        if not legal:
            break
        target, result, _prior = ordinary_search(game, rng, root, ply)
        records.append(
            {
                "state": encode_state(game.to_state(), input_encoding="kalah_v3"),
                "player": game.current_player,
                "move_index": ply,
                "legal_moves": legal,
                "policy": target,
                "search_value": result.q_value,
            }
        )
        move = sample_move(target, legal, rng)
        root = result.child_for_action(move)
        if not game.move(game.pit_index(move)):
            fail(f"illegal sampled move {seed}:{index}:{ply}")
    if not game.over():
        fail(f"unterminated game {seed}:{index}")
    winner = game.winner
    trajectory = trajectory_hash_for_encoded_states(
        [row["state"] for row in records], winner=winner
    )
    profile = build_search_profile(
        kind="self_play",
        player_mode="puct",
        simulations=SIMULATIONS,
        c_puct=1.25,
        search_options={
            "fpu_mode": "zero",
            "reuse_subtree": True,
            "normalize_values": False,
            "root_policy_mode": "visit_count",
            "tactical_root_bias": 0.0,
            "root_temperature": 0.0,
        },
    )
    for row in records:
        row["value"] = derive_self_play_value_target(
            outcome_value=outcome_for_player(winner, row["player"]),
            search_value=row.pop("search_value"),
            move_index=row["move_index"],
        )
    return index, [
        {
            **row,
            "winner": winner,
            "teacher_source": "puct",
            "policy_target_mode": "default",
            "policy_target_actual_mode": "default",
            "value_target_mode": "default",
            "search_profile": profile,
            "search_profile_hash": profile["hash"],
            "teacher_search_profile": profile,
            "teacher_search_profile_hash": profile["hash"],
            "policy_target_noise_mode": "noisy",
            "action_sampling_noise_enabled": row["move_index"] < 10,
            "target_dirichlet_epsilon": 0.3 if row["move_index"] < 10 else 0.0,
            "sampling_dirichlet_epsilon": 0.3 if row["move_index"] < 10 else 0.0,
            "simulations": SIMULATIONS,
            "dirichlet_alpha": 0.3 if row["move_index"] < 10 else 0.0,
            "dirichlet_epsilon_for_sampling": 0.3 if row["move_index"] < 10 else 0.0,
            "dirichlet_epsilon_for_target": 0.3 if row["move_index"] < 10 else 0.0,
            "game_index": index,
            "game_completed": True,
            "game_length": len(records),
            "trajectory_hash": trajectory,
        }
        for row in records
    ]


def ordinary_search(
    game: KalahGame, rng: random.Random, root: Node | None, ply: int
) -> tuple[list[float], Node, list[float]]:
    if _EVALUATOR is None:
        fail("worker evaluator not initialized")
    engine = PUCT(
        _EVALUATOR,
        SIMULATIONS,
        1.25,
        rng,
        root=root,
        fpu_mode="zero",
        reuse_subtree=True,
        normalize_values=False,
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
    )
    visits, result = engine.run(
        game,
        dirichlet_alpha=0.3 if ply < 10 else None,
        dirichlet_epsilon=0.3 if ply < 10 else 0.0,
    )
    prior = engine.root_summary()["root_prior_telemetry"]["after"]
    if prior is None:
        fail("missing root prior")
    return (
        build_policy_target(
            visits,
            legal_moves=game.possible_moves(),
            temperature=1.0 if ply < 10 else 0.1,
        ),
        result,
        list(prior),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def generate_seed(seed: int, output: Path) -> list[dict[str, Any]]:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=WORKERS, initializer=init_worker, initargs=(str(TARGET),)
    ) as executor:
        games = list(
            executor.map(
                generate_game, ((seed, index) for index in range(GAMES)), chunksize=1
            )
        )
    games.sort()
    rows = [row for _index, game in games for row in game]
    write_jsonl(output / "generated" / "ordinary_reused.jsonl", rows)
    return rows


def train_population(
    rows: list[dict[str, Any]], seed: int
) -> tuple[list[int], list[int], str]:
    train_games, validation_games = game_split(rows, seed)
    train_set, validation_set = set(train_games), set(validation_games)
    if train_set & validation_set:
        fail("game split overlap")
    indexes = [
        index for index, row in enumerate(rows) if row["game_index"] in train_set
    ]
    return (
        indexes,
        validation_games,
        canonical_sha(
            {"train_games": train_games, "validation_games": validation_games}
        ),
    )


def single_plan(pair_index: int, side: str, population: list[int]) -> list[int]:
    if len(population) < EXAMPLES:
        fail(f"pair{pair_index} {side} has fewer than {EXAMPLES} eligible train rows")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [PLAN_SEED, pair_index, PAIRS[pair_index - 1][0 if side == "A" else 1]]
        )
    )
    return [int(index) for index in rng.permutation(np.asarray(population))[:EXAMPLES]]


def pair_plan(
    a_order: list[int], b_order: list[int], pair_index: int
) -> list[list[int]]:
    if len(a_order) != EXAMPLES or len(b_order) != EXAMPLES:
        fail("single-lane ordering length mismatch")
    plan = []
    rng = np.random.default_rng(
        np.random.SeedSequence([PLAN_SEED, pair_index, 0x4D4958])
    )
    for batch in range(16):
        # pair_mix contains the first 4096 entries of each frozen single order.
        indexes = list(range(batch * 256, (batch + 1) * 256)) + list(
            range(4096 + batch * 256, 4096 + (batch + 1) * 256)
        )
        if len(indexes) != BATCH_SIZE:
            fail("pair mix batch size mismatch")
        plan.append([int(index) for index in rng.permutation(indexes)])
    return plan


def plans_for_pair(
    pair_index: int, rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> tuple[dict[str, tuple[list[dict[str, Any]], list[list[int]]]], dict[str, Any]]:
    population_a, validation_a, split_a = train_population(
        rows_a, PAIRS[pair_index - 1][0]
    )
    population_b, validation_b, split_b = train_population(
        rows_b, PAIRS[pair_index - 1][1]
    )
    a_order, b_order = (
        single_plan(pair_index, "A", population_a),
        single_plan(pair_index, "B", population_b),
    )
    lanes = {
        "single_a": (
            rows_a,
            [
                a_order[offset : offset + BATCH_SIZE]
                for offset in range(0, EXAMPLES, BATCH_SIZE)
            ],
        ),
        "single_b": (
            rows_b,
            [
                b_order[offset : offset + BATCH_SIZE]
                for offset in range(0, EXAMPLES, BATCH_SIZE)
            ],
        ),
        "pair_mix": (
            [rows_a[index] for index in a_order[:4096]]
            + [rows_b[index] for index in b_order[:4096]],
            pair_plan(a_order, b_order, pair_index),
        ),
    }
    plan_manifest = {
        name: {
            "rows": len(source),
            "batches": indexes,
            "sha256": canonical_sha(indexes),
        }
        for name, (source, indexes) in lanes.items()
    }
    return lanes, {
        "split_sha256": {"a": split_a, "b": split_b},
        "validation_games": {"a": validation_a, "b": validation_b},
        "plans": plan_manifest,
    }


def policy(state: dict[str, torch.Tensor], rows: list[dict[str, Any]]) -> np.ndarray:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    x = torch.tensor(np.asarray([row["state"] for row in rows], dtype=np.float32))
    mask = legal_mask_matrix_for_encoded_states(x.numpy())
    with torch.no_grad():
        logits, _ = model(x)
    logits = logits.numpy().astype(np.float64)
    logits[~mask.astype(bool)] = -1e9
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def train_plan(
    rows: list[dict[str, Any]],
    batches: list[list[int]],
    a16: dict[str, torch.Tensor],
    adam: dict[str, Any],
    parent: dict[str, torch.Tensor],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]]:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(copy.deepcopy(a16))
    apply_trainable_scope(model, "policy_adapter_only")
    optimizer = replay.load_isolated_optimizer(model, adam)
    parent_model = new_model(torch.device("cpu"))
    parent_model.load_state_dict(parent)
    for parameter in parent_model.parameters():
        parameter.requires_grad_(False)
    snapshots, optimizers = {}, {}
    model.train()
    for step, indexes in enumerate(batches, 1):
        batch = _batch(rows, np.asarray(indexes, dtype=np.int64), torch.device("cpu"))
        target = mixed_policy_target(
            batch["p"], incumbent_policy_batch(parent_model, batch), batch["mask"], BETA
        )
        policy_loss, value_loss = _losses(model, {**batch, "p": target})
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        if step in STEPS:
            snapshots[step] = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            optimizers[step] = copy.deepcopy(optimizer.state_dict())
    if replay.optimizer_state_sha256(adam) != ADAM_SHA:
        fail("pristine optimizer changed")
    return snapshots, optimizers


def metric_row(
    state: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    parent_policy: np.ndarray,
    initial: dict[str, torch.Tensor],
) -> dict[str, Any]:
    candidate, search = (
        policy(state, rows),
        np.asarray([row["policy"] for row in rows], dtype=np.float64),
    )
    adapter_delta = torch.cat(
        [(state[key] - initial[key]).reshape(-1) for key in ADAPTER_KEYS]
    )
    return {
        "ce_raw_reused_target": float(np.mean(_cross_entropy(candidate, search))),
        "ce_p1": float(np.mean(_cross_entropy(candidate, parent_policy))),
        "ce_beta095": float(
            np.mean(_cross_entropy(candidate, 0.05 * search + 0.95 * parent_policy))
        ),
        "legal_policy_l1_vs_p1": float(
            np.abs(candidate - parent_policy).sum(axis=1).mean()
        ),
        "js_vs_p1": float(js(candidate, parent_policy).mean()),
        "top1_disagreement": float(
            np.mean(np.argmax(candidate, axis=1) != np.argmax(parent_policy, axis=1))
        ),
        "adapter_norm": float(
            torch.linalg.vector_norm(
                torch.cat([state[key].reshape(-1) for key in ADAPTER_KEYS])
            )
        ),
        "delta_from_a16": float(torch.linalg.vector_norm(adapter_delta)),
        "inherited_parameters_bit_identical": all(
            torch.equal(state[key], initial[key])
            for key in state
            if key not in ADAPTER_KEYS
        ),
    }


def metrics(
    snapshots: dict[int, dict[str, torch.Tensor]],
    rows: list[dict[str, Any]],
    parent: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
) -> dict[str, Any]:
    parent_policy = policy(parent, rows)
    result = {"initial": metric_row(initial, rows, parent_policy, initial)}
    result.update(
        {
            str(step): metric_row(state, rows, parent_policy, initial)
            for step, state in snapshots.items()
        }
    )
    for step in ("initial", "1", "4", "16"):
        result[f"CE_beta095_{step}"] = result[step]["ce_beta095"]
        result[f"CE_raw_reused_target_{step}"] = result[step]["ce_raw_reused_target"]
    result["beta_ce_improvement"] = (
        result["initial"]["ce_beta095"] - result["16"]["ce_beta095"]
    )
    result["raw_reused_target_ce_improvement"] = (
        result["initial"]["ce_raw_reused_target"] - result["16"]["ce_raw_reused_target"]
    )
    return result


def adapter_vector(
    state: dict[str, torch.Tensor], initial: dict[str, torch.Tensor]
) -> torch.Tensor:
    return torch.cat(
        [(state[key] - initial[key]).reshape(-1) for key in ADAPTER_KEYS]
    ).double()


def full_gradient(
    rows: list[dict[str, Any]],
    a16: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
) -> torch.Tensor:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(a16)
    apply_trainable_scope(model, "policy_adapter_only")
    parent_model = new_model(torch.device("cpu"))
    parent_model.load_state_dict(parent)
    for parameter in parent_model.parameters():
        parameter.requires_grad_(False)
    total, count = None, 0
    for offset in range(0, len(rows), BATCH_SIZE):
        indexes = np.arange(offset, min(offset + BATCH_SIZE, len(rows)), dtype=np.int64)
        batch = _batch(rows, indexes, torch.device("cpu"))
        target = mixed_policy_target(
            batch["p"], incumbent_policy_batch(parent_model, batch), batch["mask"], BETA
        )
        policy_loss, value_loss = _losses(model, {**batch, "p": target})
        model.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        gradient = torch.cat(
            [
                dict(model.named_parameters())[key].grad.detach().cpu().reshape(-1)
                for key in ADAPTER_KEYS
            ]
        ).double()
        total = (
            gradient * len(indexes)
            if total is None
            else total + gradient * len(indexes)
        )
        count += len(indexes)
    if total is None:
        fail("empty gradient population")
    return total / count


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if torch.linalg.vector_norm(left) == 0 or torch.linalg.vector_norm(right) == 0:
        fail("zero diagnostic vector")
    return float(
        torch.dot(left, right)
        / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    )


def diversity(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> dict[str, Any]:
    states_a, states_b = (
        {canonical_sha(row["state"]) for row in rows_a},
        {canonical_sha(row["state"]) for row in rows_b},
    )
    trajectories_a, trajectories_b = (
        {row["trajectory_hash"] for row in rows_a},
        {row["trajectory_hash"] for row in rows_b},
    )
    outcomes_a, outcomes_b = (
        Counter(row["winner"] for row in rows_a),
        Counter(row["winner"] for row in rows_b),
    )

    def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in rows).items()))

    outcome_difference = sum(
        abs(outcomes_a[key] / len(rows_a) - outcomes_b[key] / len(rows_b))
        for key in set(outcomes_a) | set(outcomes_b)
    )
    return {
        "exact_state_hash_overlap": len(states_a & states_b),
        "state_jaccard": len(states_a & states_b) / len(states_a | states_b),
        "game_trajectory_hash_overlap": len(trajectories_a & trajectories_b),
        "outcome_distribution_l1": outcome_difference,
        "move_index_distribution": {
            "a": distribution(rows_a, "move_index"),
            "b": distribution(rows_b, "move_index"),
        },
        "value_target_distribution": {
            "a": distribution(rows_a, "value"),
            "b": distribution(rows_b, "value"),
        },
    }


def seal_suites(
    output: Path,
    registry: dict[str, registry_module.ConsumedSuite],
    replays: dict[int, list[dict[str, Any]]],
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    registry_module.validate(registry)
    used, old_prefixes = (
        registry_module.final_keys(registry),
        registry_module.prefix_keys(registry),
    )
    replay_states = set().union(
        *(set(tuple(row["state"]) for row in rows) for rows in replays.values())
    )
    universe = [
        entry
        for entry in pr249.all_openings()
        if tuple(encode_state(entry["state"], input_encoding="kalah_v3"))
        not in replay_states
    ]
    paths, selected_prefixes, manifest = (
        {},
        set(),
        {"consumed": registry_module.manifest(registry), "newly_consumed": {}},
    )
    for label, seed in SUITE_SEEDS.items():
        selected = suites.select_diverse(
            [
                entry
                for entry in universe
                if suites.canonical_key(entry["state"]) not in used
                and not (
                    pr251.prefix_keys([entry]) & (old_prefixes | selected_prefixes)
                )
            ],
            128,
            seed,
        )
        keys, prefixes = pr249.suite_keys(selected), pr251.prefix_keys(selected)
        if (
            len(keys) != 128
            or keys & used
            or prefixes & (old_prefixes | selected_prefixes)
        ):
            fail(f"suite overlap {label}")
        path = output / "suites" / f"suite_{label}.jsonl"
        suites.write_suite_jsonl(selected, str(path))
        paths[label] = path
        manifest["newly_consumed"][label] = {
            "seed": seed,
            "sha256": sha256_file(path),
            "openings": 128,
            "status": "consumed",
        }
        # Persist consumption before selecting the next suite or running any arena.
        (output / "suite_registry.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        used |= keys
        selected_prefixes |= prefixes
    report = {
        "passed": True,
        "registry": registry_module.manifest(registry),
        "suites": manifest["newly_consumed"],
    }
    for label, path in paths.items():
        entries = suites.load_suite_jsonl(str(path))
        keys, prefixes = pr249.suite_keys(entries), pr251.prefix_keys(entries)
        report[label] = {
            "final_overlap_registry": len(keys & registry_module.final_keys(registry)),
            "prefix_overlap_registry": len(
                prefixes & registry_module.prefix_keys(registry)
            ),
            "replay_state_overlap": sum(
                tuple(encode_state(row["state"], input_encoding="kalah_v3"))
                in replay_states
                for row in entries
            ),
        }
        if any(report[label].values()):
            fail(f"suite preflight overlap {label}")
    return paths, manifest, report


def evaluate(
    candidates: dict[str, Path],
    suites_paths: dict[str, Path],
    output: Path,
    context: str,
) -> dict[str, Any]:
    result = {}
    for label, suite in suites_paths.items():
        control = isolation.arena_records(
            output / "arena" / context / label,
            replay.P1_CHECKPOINT.parent / "artifact",
            replay.P1_CHECKPOINT.parent / "artifact",
            context,
            "p1_control",
            WORKERS,
            suite,
        )
        result[label] = {}
        for lane, candidate in candidates.items():
            records = isolation.arena_records(
                output / "arena" / context / label,
                candidate,
                replay.P1_CHECKPOINT.parent / "artifact",
                context,
                lane,
                WORKERS,
                suite,
            )
            result[label][lane] = {
                "effect": paired_opening_candidate_effect(records, control),
                "wdl": pr244.win_draw_loss(records),
            }
    return result


def group_pair_evaluation(raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Restore lane names after arena artifacts are made globally unique."""
    return {
        label: {
            f"pair{pair_index}": {
                lane: raw[label][f"pair{pair_index}_{lane}"]
                for lane in ("single_a", "single_b", "pair_mix")
            }
            for pair_index in range(1, 6)
        }
        for label in SUITE_SEEDS
    }


def analyze(evaluation: dict[str, Any]) -> dict[str, Any]:
    per_pair, rng = {}, np.random.default_rng(42)
    for pair_index, _pair in enumerate(PAIRS, 1):
        name = f"pair{pair_index}"
        suite_values, strengths = (
            [],
            {lane: [] for lane in ("single_a", "single_b", "pair_mix")},
        )
        for label in SUITE_SEEDS:
            values = evaluation[label][name]
            diff_a = paired_effect_difference(
                values["pair_mix"]["effect"], values["single_a"]["effect"]
            )
            diff_b = paired_effect_difference(
                values["pair_mix"]["effect"], values["single_b"]["effect"]
            )
            delta = 0.5 * (
                np.asarray(list(diff_a["per_opening_effect"].values()))
                + np.asarray(list(diff_b["per_opening_effect"].values()))
            )
            suite_values.append(delta)
            for lane in strengths:
                strengths[lane].append(
                    values[lane]["effect"]["paired_candidate_effect"]
                )
        single_a, single_b = (
            float(np.mean(strengths["single_a"])),
            float(np.mean(strengths["single_b"])),
        )
        per_pair[name] = {
            "delta": float(np.concatenate(suite_values).mean()),
            "suite_deltas": {
                label: float(value.mean())
                for label, value in zip(SUITE_SEEDS, suite_values, strict=True)
            },
            "single_a_strength": single_a,
            "single_b_strength": single_b,
            "single_strength_mean": 0.5 * (single_a + single_b),
            "single_strength_spread": abs(single_a - single_b),
            "pair_mix_strength": float(np.mean(strengths["pair_mix"])),
        }
    deltas = np.asarray([per_pair[f"pair{i}"]["delta"] for i in range(1, 6)])
    draws = []
    for _ in range(10_000):
        samples = []
        for pair_index in rng.integers(1, 6, 5):
            label = list(SUITE_SEEDS)[rng.integers(0, 3)]
            values = evaluation[label][f"pair{pair_index}"]
            diff_a, diff_b = (
                paired_effect_difference(
                    values["pair_mix"]["effect"], values["single_a"]["effect"]
                ),
                paired_effect_difference(
                    values["pair_mix"]["effect"], values["single_b"]["effect"]
                ),
            )
            a, b = (
                np.asarray(list(diff_a["per_opening_effect"].values())),
                np.asarray(list(diff_b["per_opening_effect"].values())),
            )
            samples.append(
                float(
                    (
                        rng.choice(a, len(a), replace=True).mean()
                        + rng.choice(b, len(b), replace=True).mean()
                    )
                    / 2
                )
            )
        draws.append(float(np.mean(samples)))
    individual = np.asarray(
        [
            value
            for pair in per_pair.values()
            for value in (pair["single_a_strength"], pair["single_b_strength"])
        ]
    )
    mixed = np.asarray([pair["pair_mix_strength"] for pair in per_pair.values()])
    return {
        "per_pair": per_pair,
        "primary": {
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "sd": float(deltas.std(ddof=1)),
            "min": float(deltas.min()),
            "max": float(deltas.max()),
            "positive_pairs": int((deltas > 0).sum()),
            "hierarchical_pair_suite_opening_ci": {
                "lower_95": float(np.quantile(draws, 0.025)),
                "upper_95": float(np.quantile(draws, 0.975)),
            },
        },
        "absolute_strength_variance": {
            "individual_single_replay_effects": float(individual.var(ddof=1)),
            "pair_mix_effects": float(mixed.var(ddof=1)),
        },
    }


def classify(
    primary: dict[str, Any], training: dict[str, Any], shallow: dict[str, Any] | None
) -> str:
    values, summary = (
        [primary["per_pair"][f"pair{i}"]["delta"] for i in range(1, 6)],
        primary["primary"],
    )
    learning = all(
        lane["metrics"]["beta_ce_improvement"] > 0 for lane in training.values()
    )
    success = (
        summary["mean_delta"] > 0
        and summary["hierarchical_pair_suite_opening_ci"]["lower_95"] > 0
        and summary["positive_pairs"] >= 4
        and min(values) >= -0.02
        and learning
    )
    shallow_harm = shallow is not None and (
        shallow["primary"]["mean_delta"] < -0.02
        or sum(
            value < -0.02
            for value in [shallow["per_pair"][f"pair{i}"]["delta"] for i in range(1, 6)]
        )
        >= 2
    )
    if success:
        return (
            "aggregation_high_budget_gain_with_shallow_harm"
            if shallow_harm
            else "two_replay_aggregation_improves_strength"
        )
    if summary["mean_delta"] > 0:
        return "aggregation_improves_mean_but_is_variable"
    variance = primary["absolute_strength_variance"]
    return (
        "aggregation_reduces_variance_only"
        if variance["pair_mix_effects"] < variance["individual_single_replay_effects"]
        else "two_replay_aggregation_not_helpful"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr258_two_replay_aggregation")
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--run-shallow", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    registry = registry_module.load(
        args.workdir
    )  # Must pass before any new replay or suite is created.
    snapshot = torch.load(replay.A16_SNAPSHOT, map_location="cpu", weights_only=False)
    a16, adam = replay.immutable_initial_state(snapshot)
    if (
        sha256_file(replay.A16_SNAPSHOT) != A16_SHA
        or replay.optimizer_state_sha256(adam) != ADAM_SHA
        or sha256_file(replay.P1_CHECKPOINT) != P1_SHA
        or sha256_file(TARGET) != TARGET_SHA
    ):
        fail("frozen artifact mismatch")
    args.workdir.mkdir(parents=True, exist_ok=True)
    blocked, groups = replay.exclusion_hashes(pr249.CANONICAL_SUITE)
    consumed = set().union(
        *(replay.canonical_arena_hashes(spec.path) for spec in registry.values())
    )
    blocked |= consumed
    groups = {**groups, "consumed_evaluation_openings": consumed}
    p1 = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1, replay.P1_CHECKPOINT)
    parent = {
        key: value.detach().cpu().clone() for key, value in p1.state_dict().items()
    }
    replays, replay_manifest = {}, {}
    for seed in SEEDS:
        raw = generate_seed(seed, args.workdir / f"seed{seed}")
        eligible, exclusions = replay.filter_rows(raw, blocked, groups)
        replays[seed] = eligible
        replay_manifest[str(seed)] = {
            "raw_rows": len(raw),
            "eligible_rows": len(eligible),
            "replay_sha256": sha256_file(
                args.workdir / f"seed{seed}" / "generated" / "ordinary_reused.jsonl"
            ),
            "state_outcome_sha256": canonical_sha(
                [
                    {key: value for key, value in row.items() if key != "policy"}
                    for row in eligible
                ]
            ),
            "exclusions": exclusions,
        }
    (args.workdir / "frozen_replays.json").write_text(
        json.dumps(replay_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pair_lanes, plan_manifests = {}, {}
    for pair_index, (seed_a, seed_b) in enumerate(PAIRS, 1):
        pair_name = f"pair{pair_index}"
        pair_lanes[pair_name], plan_manifests[pair_name] = plans_for_pair(
            pair_index, replays[seed_a], replays[seed_b]
        )
    # All deterministic sample evidence is committed before any model is trained.
    frozen_plans = {"replays": replay_manifest, "sample_plans": plan_manifests}
    (args.workdir / "frozen_replays_and_plans.json").write_text(
        json.dumps(frozen_plans, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training, candidates, diagnostics = {}, {}, {}
    for pair_index, (seed_a, seed_b) in enumerate(PAIRS, 1):
        pair_name, lanes = f"pair{pair_index}", pair_lanes[f"pair{pair_index}"]
        plan_manifest = plan_manifests[pair_name]
        training[pair_name] = {}
        candidates[pair_name] = {}
        diagnostics[pair_name] = {
            "replay_diversity": diversity(replays[seed_a], replays[seed_b])
        }
        gradients = {
            side: full_gradient(replays[seed], a16, parent)
            for side, seed in (("a", seed_a), ("b", seed_b))
        }
        g_mean = 0.5 * (gradients["a"] + gradients["b"])
        diagnostics[pair_name]["gradient_variance"] = {
            "cosine_gA_gB": cosine(gradients["a"], gradients["b"]),
            "norm_gA": float(torch.linalg.vector_norm(gradients["a"])),
            "norm_gB": float(torch.linalg.vector_norm(gradients["b"])),
        }
        first_run = {}
        for lane, (source, batches) in lanes.items():
            snapshots, optimizers = train_plan(source, batches, a16, adam, parent)
            metric_rows = (
                source if lane != "pair_mix" else replays[seed_a] + replays[seed_b]
            )
            lane_result = {
                "plan": plan_manifest["plans"][lane],
                "metrics": metrics(snapshots, metric_rows, parent, a16),
                "checkpoints": {
                    str(step): {
                        "model_sha256": contract.state_sha256(snapshots[step]),
                        "optimizer_sha256": replay.optimizer_state_sha256(
                            optimizers[step]
                        ),
                    }
                    for step in STEPS
                },
            }
            checkpoint = args.workdir / "train" / pair_name / lane / "step_0016.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": snapshots[16], "optimizer": optimizers[16]}, checkpoint
            )
            candidates[pair_name][lane] = export(
                snapshots[16], checkpoint.with_suffix(""), f"pr258_{pair_name}_{lane}"
            )
            training[pair_name][lane] = lane_result
            first_run[lane] = lane_result
            if lane == "pair_mix":
                diagnostics[pair_name]["gradient_variance"][
                    "pair_mix_step1_update_cosine_negative_g_mean"
                ] = cosine(adapter_vector(snapshots[1], a16), -g_mean)
        # One pair is rerun forward and reverse to detect mutable model or Adam state.
        if pair_index == 1:
            for order in (
                ("single_a", "single_b", "pair_mix"),
                ("pair_mix", "single_b", "single_a"),
            ):
                for lane in order:
                    snapshots, optimizers = train_plan(*lanes[lane], a16, adam, parent)
                    if (
                        contract.state_sha256(snapshots[16])
                        != first_run[lane]["checkpoints"]["16"]["model_sha256"]
                        or replay.optimizer_state_sha256(optimizers[16])
                        != first_run[lane]["checkpoints"]["16"]["optimizer_sha256"]
                    ):
                        fail("repeated-lane or lane-order dependence")
    frozen = {
        "schema": "alphazero_lite_pr258_two_replay_aggregation_v1",
        "ordering": [
            "registry_validation",
            "all_replays",
            "replay_and_plan_freeze",
            "all_candidates",
            "model_freeze",
            "VWX",
            "preflight",
            "arena",
        ],
        "artifacts": {
            "a16": A16_SHA,
            "adam": ADAM_SHA,
            "p1": P1_SHA,
            "evaluator": TARGET_SHA,
        },
        "replays": replay_manifest,
        "training": training,
        "diagnostics": diagnostics,
    }
    frozen["candidate_model_sha256"] = canonical_sha(
        {
            pair: {
                lane: value["checkpoints"]["16"]["model_sha256"]
                for lane, value in lanes.items()
            }
            for pair, lanes in training.items()
        }
    )
    gradient_cosines = np.asarray(
        [
            diagnostics[f"pair{pair_index}"]["gradient_variance"]["cosine_gA_gB"]
            for pair_index in range(1, 6)
        ]
    )
    diagnostics["gradient_cosine_distribution"] = {
        "mean": float(gradient_cosines.mean()),
        "median": float(np.median(gradient_cosines)),
        "min": float(gradient_cosines.min()),
        "max": float(gradient_cosines.max()),
        "values": gradient_cosines.tolist(),
    }
    (args.workdir / "frozen_candidates.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths, suite_manifest, preflight = seal_suites(args.workdir, registry, replays)
    frozen["suite_manifest"], frozen["preflight"] = suite_manifest, preflight
    preflight_path = args.workdir / "preflight_audit.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen["suite_manifest_sha256"] = canonical_sha(suite_manifest)
    frozen["preflight_sha256"] = sha256_file(preflight_path)
    frozen_path = args.workdir / "frozen_manifest.json"
    frozen_path.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen_manifest_sha256 = sha256_file(frozen_path)
    if args.freeze_only:
        return
    if not preflight["passed"]:
        fail("arena attempted before successful preflight")
    flat_candidates = {
        f"pair{i}_{lane}": candidates[f"pair{i}"][lane]
        for i in range(1, 6)
        for lane in candidates[f"pair{i}"]
    }
    raw_primary = evaluate(flat_candidates, paths, args.workdir, "1200:1200")
    primary = analyze(group_pair_evaluation(raw_primary))
    primary_path = args.workdir / "primary_1200_results.json"
    primary_path.write_text(
        json.dumps(primary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shallow = (
        analyze(
            group_pair_evaluation(
                evaluate(flat_candidates, paths, args.workdir, "384:256")
            )
        )
        if args.run_shallow
        else None
    )
    result = {
        **frozen,
        "primary_evaluation": primary,
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "shallow_evaluation": shallow,
        "compute": {
            "single_replay_games": 700,
            "pair_aggregation_games": 1400,
            "search_simulations_per_move": 384,
            "self_play_search_compute_multiplier": 2.0,
            "sgd_examples": EXAMPLES,
            "optimizer_steps": 16,
            "wall_clock_seconds": time.monotonic() - started,
        },
        "classification": classify(
            primary,
            {
                f"{pair}_{lane}": value
                for pair, lanes in training.items()
                for lane, value in lanes.items()
            },
            shallow,
        ),
    }
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
