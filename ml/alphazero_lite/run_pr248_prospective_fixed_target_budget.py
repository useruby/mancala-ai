#!/usr/bin/env python3
# ruff: noqa: E402
"""Prospectively test fixed 768 policy targets on a new seed-46 batch."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
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

from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as replay_train
from ml.alphazero_lite import (
    run_pr241_optimizer_isolation_reproduction as train_contract,
)
from ml.alphazero_lite import run_pr241_policy_target_noise_isolation as isolation
from ml.alphazero_lite import run_pr242_target_entropy_factorization as pr244
from ml.alphazero_lite.evaluation_metrics import paired_effect_difference
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    export,
    new_model,
)
from ml.alphazero_lite.run_fresh_p1_shadow_target_distillation import _frozen_diagnostic
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
from ml.alphazero_lite.train import load_checkpoint_into_model

GAMES, WORKERS, SEED, BASE, MAX_MOVES = 700, 24, 46, 384, 200
LANES = ("reused", "fresh384", "fresh768", "fresh1024")
CONTEXTS = ("384:256", "1200:1200")
TARGET_CHECKPOINT = Path(
    "/tmp/azlite_fresh_p1_parent_adapter/artifacts/step_0016/checkpoint.npz"
)
A16_SHA = "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
ADAM_SHA = "61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7"
P1_SHA = "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9"
TARGET_SHA = "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34"
SUITE_SHA = "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04"
_EVALUATOR: CheckpointEvaluator | None = None
_PLY = 0


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def worker_for(game_index: int) -> int:
    cursor = 0
    for worker, count in enumerate([30] * 4 + [29] * 20):
        if cursor <= game_index < cursor + count:
            return worker
        cursor += count
    fail(f"game index outside contract: {game_index}")


def game_rng(game_index: int) -> random.Random:
    return random.Random(SEED * 1_000_003 + game_index + worker_for(game_index) * 9_973)


def inherited_mass(root: Node | None) -> int:
    return (
        0
        if root is None
        else int(sum(child.visit_count for child in root.children.values()))
    )


def search(
    game: KalahGame,
    rng: random.Random,
    *,
    root: Node | None,
    reuse: bool,
    simulations: int,
    noisy: bool,
) -> tuple[list[float], Node, list[float]]:
    if _EVALUATOR is None:
        raise RuntimeError("worker evaluator is not initialized")
    engine = PUCT(
        _EVALUATOR,
        simulations,
        1.25,
        rng,
        root=root,
        fpu_mode="zero",
        reuse_subtree=reuse,
        normalize_values=False,
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
    )
    visits, result_root = engine.run(
        game,
        dirichlet_alpha=0.3 if noisy else None,
        dirichlet_epsilon=0.3 if noisy else 0.0,
    )
    prior = engine.root_summary()["root_prior_telemetry"]["after"]
    if prior is None:
        fail("missing root prior")
    return (
        build_policy_target(
            visits,
            legal_moves=game.possible_moves(),
            temperature=1.0 if _PLY < 10 else 0.1,
        ),
        result_root,
        [float(x) for x in prior],
    )


def init_worker(checkpoint: str) -> None:
    global _EVALUATOR
    _EVALUATOR = CheckpointEvaluator(Path(checkpoint), input_encoding="kalah_v3")


def generate_game(
    game_index: int,
) -> tuple[int, dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    global _PLY
    game, rng, reusable_root = (
        KalahGame.from_state(standard_start_state()),
        game_rng(game_index),
        None,
    )
    records: list[dict[str, Any]] = []
    for ply in range(MAX_MOVES):
        if game.over():
            break
        legal = game.possible_moves()
        if not legal:
            break
        _PLY, before, mass = ply, rng.getstate(), inherited_mass(reusable_root)
        runs = {}
        for lane, budget in (
            ("reused", BASE),
            ("fresh384", BASE),
            ("fresh768", 768),
            ("fresh1024", 1024),
        ):
            clone = random.Random()
            clone.setstate(before)
            policy, result_root, prior = search(
                game,
                clone,
                root=reusable_root if lane == "reused" else None,
                reuse=lane == "reused",
                simulations=budget,
                noisy=ply < 10,
            )
            runs[lane] = (policy, result_root, prior, clone)
        priors = [runs[lane][2] for lane in LANES]
        if not all(np.array_equal(priors[0], prior) for prior in priors[1:]):
            fail(f"root-noise mismatch game={game_index} ply={ply}")
        if ply == 0 and (mass != 0 or runs["reused"][0] != runs["fresh384"][0]):
            fail(f"ply-0 mismatch game={game_index}")
        records.append(
            {
                "state": encode_state(game.to_state(), input_encoding="kalah_v3"),
                "player": game.current_player,
                "move_index": ply,
                "legal_moves": legal,
                "search_value": runs["reused"][1].q_value,
                "policies": {lane: runs[lane][0] for lane in LANES},
                "inherited_child_visit_mass": mass,
            }
        )
        move = sample_move(runs["reused"][0], legal, runs["reused"][3])
        reusable_root = runs["reused"][1].child_for_action(move)
        if not game.move(game.pit_index(move)):
            fail(f"illegal sampled move game={game_index} ply={ply}")
        rng = runs["reused"][3]
    if not game.over():
        fail(f"unterminated game={game_index}")
    winner = game.winner
    trajectory_hash = trajectory_hash_for_encoded_states(
        [r["state"] for r in records], winner=winner
    )
    profile = build_search_profile(
        kind="self_play",
        player_mode="puct",
        simulations=BASE,
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
    views = {lane: [] for lane in LANES}
    telemetry = []
    for record in records:
        common = {
            "state": record["state"],
            "value": derive_self_play_value_target(
                outcome_value=outcome_for_player(winner, record["player"]),
                search_value=record["search_value"],
                move_index=record["move_index"],
            ),
            "player": record["player"],
            "move_index": record["move_index"],
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
            "action_sampling_noise_enabled": record["move_index"] < 10,
            "target_dirichlet_epsilon": 0.3 if record["move_index"] < 10 else 0.0,
            "sampling_dirichlet_epsilon": 0.3 if record["move_index"] < 10 else 0.0,
            "simulations": BASE,
            "dirichlet_alpha": 0.3 if record["move_index"] < 10 else 0.0,
            "dirichlet_epsilon_for_sampling": 0.3 if record["move_index"] < 10 else 0.0,
            "dirichlet_epsilon_for_target": 0.3 if record["move_index"] < 10 else 0.0,
            "legal_moves": record["legal_moves"],
            "game_index": game_index,
            "game_completed": True,
            "game_length": len(records),
            "trajectory_hash": trajectory_hash,
        }
        for lane in LANES:
            views[lane].append({**common, "policy": record["policies"][lane]})
        telemetry.append(
            {
                "game_index": game_index,
                "move_index": record["move_index"],
                "inherited_child_visit_mass": record["inherited_child_visit_mass"],
                "policies": record["policies"],
            }
        )
    return game_index, views, telemetry


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def generate(
    workdir: Path, workers: int
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(str(TARGET_CHECKPOINT),)
    ) as executor:
        completed = list(executor.map(generate_game, range(GAMES), chunksize=1))
    completed.sort(key=lambda item: item[0])
    views = {
        lane: [row for _, lane_rows, _ in completed for row in lane_rows[lane]]
        for lane in LANES
    }
    telemetry = [row for _, _, values in completed for row in values]
    for lane, rows in views.items():
        write_jsonl(workdir / "generated" / f"{lane}.jsonl", rows)
    write_jsonl(workdir / "generated" / "telemetry.jsonl", telemetry)
    return views, telemetry


def non_policy_sha(rows: list[dict[str, Any]]) -> str:
    payload = [
        {key: value for key, value in row.items() if key != "policy"} for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def target_diagnostics(
    rows: list[dict[str, Any]], telemetry: list[dict[str, Any]]
) -> dict[str, Any]:
    masses = np.asarray(
        [row["inherited_child_visit_mass"] for row in telemetry], dtype=float
    )
    edges = np.quantile(masses, [0.25, 0.5, 0.75])
    targets = {lane: [row["policies"][lane] for row in telemetry] for lane in LANES}
    result: dict[str, Any] = {}
    for left, right in (
        ("reused", "fresh384"),
        ("reused", "fresh768"),
        ("reused", "fresh1024"),
        ("fresh384", "fresh768"),
        ("fresh768", "fresh1024"),
    ):
        stats = [
            isolation.policy_stats(a, b, row["legal_moves"])
            for row, a, b in zip(rows, targets[left], targets[right], strict=True)
        ]
        groups: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
        for row, item, mass in zip(rows, stats, masses, strict=True):
            groups[f"phase:{isolation.phase(row)}"].append(item)
            groups[f"legal_move_count:{len(row['legal_moves'])}"].append(item)
            groups[
                f"root_noise_enabled:{bool(row['action_sampling_noise_enabled'])}"
            ].append(item)
            groups[
                f"inherited_visit_mass_quartile:{np.searchsorted(edges, mass, side='right') + 1}"
            ].append(item)
        result[f"{left}_vs_{right}"] = {
            "overall": isolation.summarize(stats),
            "strata": {
                key: isolation.summarize(value) for key, value in sorted(groups.items())
            },
        }
    movements: dict[str, list[float]] = defaultdict(list)
    for row, a, b, c in zip(
        rows,
        targets["fresh384"],
        targets["fresh768"],
        targets["fresh1024"],
        strict=True,
    ):
        legal = row["legal_moves"]
        top = [legal[int(np.argmax(np.asarray(policy)[legal]))] for policy in (a, b, c)]
        category = (
            "stable"
            if top[0] == top[1] == top[2]
            else "768_only_flip"
            if top[0] != top[1] and top[2] == top[0]
            else "persistent_flip"
            if top[0] != top[1] and top[2] == top[1]
            else "late_flip"
            if top[0] == top[1] and top[2] != top[1]
            else "other"
        )
        movements[category].append(
            float(
                np.abs(np.asarray(b)[legal] - np.asarray(a)[legal]).sum()
                + np.abs(np.asarray(c)[legal] - np.asarray(b)[legal]).sum()
            )
        )
    result["search_budget_movement"] = {
        key: {
            "frequency": len(movements[key]) / len(rows),
            "mean_two_transition_legal_l1": float(np.mean(movements[key]))
            if movements[key]
            else 0.0,
        }
        for key in ("stable", "768_only_flip", "persistent_flip", "late_flip", "other")
    }
    return result


def train_lanes(
    rows: dict[str, list[dict[str, Any]]],
    model: dict[str, torch.Tensor],
    optimizer: dict[str, Any],
    parent: dict[str, torch.Tensor],
    workdir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    initial, result, artifacts = replay_train.optimizer_state_sha256(optimizer), {}, {}
    targets = {lane: [row["policy"] for row in values] for lane, values in rows.items()}
    states: dict[
        str, tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]]
    ] = {}
    for lane in LANES:
        snapshots, optimizers, invocation = train_contract.run_lane(
            rows[lane], model, optimizer, parent
        )
        states[lane] = snapshots, optimizers
        result[lane] = {
            "optimizer_invocation": invocation,
            "metrics": replay_train.metrics(
                rows[lane], snapshots, parent, model, copy.deepcopy(optimizer)
            ),
            "checkpoints": {},
            "cross_target_ce": {},
        }
        for step, state in snapshots.items():
            output = workdir / "train" / lane / f"step_{step:04d}"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": state, "optimizer": optimizers[step]},
                output.with_suffix(".pt"),
            )
            result[lane]["checkpoints"][str(step)] = train_contract.checkpoint_record(
                state, optimizers[step]
            )
            result[lane]["cross_target_ce"][str(step)] = isolation.all_target_metrics(
                rows[lane], state, targets
            )
            if step == 16:
                artifacts[lane] = export(state, output, f"pr248_{lane}")
    repeated = train_contract.repeated_lane_check(
        "fresh768", rows["fresh768"], model, optimizer, parent
    )
    reordered = {}
    for lane in ("fresh1024", "fresh768"):
        snapshots, optimizers, _ = train_contract.run_lane(
            rows[lane], model, optimizer, parent
        )
        reordered[lane] = {
            str(step): train_contract.checkpoint_record(
                snapshots[step], optimizers[step]
            )
            for step in replay_train.STEPS
        }
    order_independent = all(
        reordered[lane] == result[lane]["checkpoints"] for lane in reordered
    )
    if (
        not repeated
        or not order_independent
        or replay_train.optimizer_state_sha256(optimizer) != initial
    ):
        fail("optimizer contamination, repeated identity, or order dependence")
    result["optimizer_invariants"] = {
        "initial_sha256": initial,
        "pristine_unchanged": replay_train.optimizer_state_sha256(optimizer) == initial,
        "repeated_fresh768": repeated,
        "fresh768_fresh1024_order_independent": order_independent,
    }
    return result, artifacts


def arena_summary(
    workdir: Path, artifacts: dict[str, Path], p1: Path, suite: Path, workers: int
) -> dict[str, Any]:
    controls = {
        context: isolation.arena_records(
            workdir / "arena", p1, p1, context, "p1_control", workers, suite
        )
        for context in CONTEXTS
    }
    result, effects = {}, {}
    for lane in LANES:
        result[lane], effects[lane] = {}, {}
        for context in CONTEXTS:
            records = isolation.arena_records(
                workdir / "arena", artifacts[lane], p1, context, lane, workers, suite
            )
            effect = isolation.paired_opening_candidate_effect(
                records, controls[context]
            )
            effects[lane][context] = effect
            result[lane][context] = {
                "effect": effect["paired_candidate_effect"],
                "ci": effect["opening_bootstrap_ci"],
                "seat_effects": {"p0": effect["p0_effect"], "p1": effect["p1_effect"]},
                "win_draw_loss": pr244.win_draw_loss(records),
            }
    result["paired_contrasts"] = {
        f"{left}_minus_{right}": {
            context: paired_effect_difference(
                effects[left][context], effects[right][context]
            )
            for context in CONTEXTS
        }
        for left, right in (
            ("fresh768", "fresh384"),
            ("reused", "fresh384"),
            ("fresh768", "fresh1024"),
        )
    }
    return result


def classify(training: dict[str, Any], arena: dict[str, Any]) -> str:
    deep = "1200:1200"
    contrast = arena["paired_contrasts"]
    fresh_pass = (
        contrast["fresh768_minus_fresh384"][deep]["opening_bootstrap_ci"]["lower_95"]
        > 0
    )
    reused_pass = (
        contrast["reused_minus_fresh384"][deep]["opening_bootstrap_ci"]["lower_95"] > 0
    )
    fixed1024_pass = (
        arena["fresh1024"][deep]["effect"] > arena["fresh384"][deep]["effect"]
    )
    fit = training["fresh768"]["metrics"]["16"]["fit_fraction"] >= 0.25
    safe = arena["fresh768"][deep]["ci"]["lower_95"] >= -0.03 or (
        arena["fresh768"][deep]["ci"]["lower_95"]
        <= 0
        <= arena["fresh768"][deep]["ci"]["upper_95"]
    )
    if fresh_pass and reused_pass and fit and safe:
        return (
            "fixed_compute_generalizes_not_768_specific"
            if fixed1024_pass
            else "narrow_768_budget_window_replicates"
        )
    if fresh_pass and reused_pass:
        return "prospective_fixed768_replica"
    if reused_pass and not fresh_pass and not fixed1024_pass:
        return "fixed_target_search_not_prospectively_stable"
    if reused_pass and not fresh_pass:
        return "fixed768_replay_specific"
    return "target_depth_mechanism_fails_new_batch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr248_prospective_fixed_target_budget"),
    )
    parser.add_argument(
        "--canonical-suite",
        type=Path,
        default=Path("/tmp/azlite_opening_suite/medium_eval.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--skip-arena", action="store_true")
    args = parser.parse_args()
    if (
        args.workers != WORKERS
        or sha256_file(TARGET_CHECKPOINT) != TARGET_SHA
        or sha256_file(args.canonical_suite) != SUITE_SHA
    ):
        fail("target evaluator, canonical suite, or worker mismatch")
    snapshot = torch.load(
        replay_train.A16_SNAPSHOT, map_location="cpu", weights_only=False
    )
    model, optimizer = replay_train.immutable_initial_state(snapshot)
    if (
        sha256_file(replay_train.A16_SNAPSHOT) != A16_SHA
        or sha256_file(replay_train.P1_CHECKPOINT) != P1_SHA
        or replay_train.optimizer_state_sha256(optimizer) != ADAM_SHA
    ):
        fail("A16 snapshot, P1 checkpoint, or pristine Adam mismatch")
    generated, all_telemetry = generate(args.workdir, args.workers)
    blocked, groups = replay_train.exclusion_hashes(args.canonical_suite)
    eligible, exclusions = {}, {}
    for lane in LANES:
        eligible[lane], exclusions[lane] = replay_train.filter_rows(
            generated[lane], blocked, groups
        )
    invariants = isolation.assert_views(eligible["reused"], eligible)
    hashes = {lane: non_policy_sha(eligible[lane]) for lane in LANES}
    if len(set(hashes.values())) != 1 or any(
        exclusions[lane] != exclusions["reused"] for lane in LANES
    ):
        fail("replay view state/outcome or exclusion mismatch")
    keys = {(row["game_index"], row["move_index"]) for row in eligible["reused"]}
    telemetry = [
        row for row in all_telemetry if (row["game_index"], row["move_index"]) in keys
    ]
    if len(telemetry) != len(eligible["reused"]):
        fail("telemetry mismatch")
    p1 = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1, replay_train.P1_CHECKPOINT)
    parent = {
        name: value.detach().cpu().clone() for name, value in p1.state_dict().items()
    }
    training, artifacts = train_lanes(eligible, model, optimizer, parent, args.workdir)
    result: dict[str, Any] = {
        "schema": "pr248_prospective_fixed_target_budget_v1",
        "classification": "inconclusive",
        "frozen_artifacts": {
            "a16_snapshot_sha256": A16_SHA,
            "initial_optimizer_sha256": ADAM_SHA,
            "p1_checkpoint_sha256": P1_SHA,
            "target_evaluator_sha256": TARGET_SHA,
            "canonical_suite_sha256": SUITE_SHA,
        },
        "generation": {
            "games": GAMES,
            "workers": WORKERS,
            "seed": SEED,
            "gameplay_simulations": BASE,
            "target_simulations": {"fresh384": 384, "fresh768": 768, "fresh1024": 1024},
            "replay_sha256": {
                lane: sha256_file(args.workdir / "generated" / f"{lane}.jsonl")
                for lane in LANES
            },
            "policy_excluded_state_outcome_sha256": hashes,
        },
        "exclusions": exclusions["reused"],
        "invariants": invariants,
        "target_diagnostics": target_diagnostics(eligible["reused"], telemetry),
        "training": training,
        "frozen_40_40": {
            lane: _frozen_diagnostic(
                artifact,
                replay_train.P1_CHECKPOINT.parent / "artifact",
                read_jsonl(
                    Path("/tmp/azlite_onpolicy_shadow_replay/p1_reference.jsonl")
                ),
            )
            for lane, artifact in artifacts.items()
        },
    }
    if not args.skip_arena:
        result["canonical_arena"] = arena_summary(
            args.workdir,
            artifacts,
            replay_train.P1_CHECKPOINT.parent / "artifact",
            args.canonical_suite,
            args.workers,
        )
        result["classification"] = classify(training, result["canonical_arena"])
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
