#!/usr/bin/env python3
# ruff: noqa: E402
"""Prospectively replicate PR #246's effective policy-target compute result.

One authoritative reused-tree A16 trajectory is generated for each game.  Two
fresh, matched-RNG searches produce policy-only replay views and never consume
the gameplay RNG or influence its selected moves.
"""

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
from ml.alphazero_lite import run_pr241_optimizer_isolation_reproduction as pr245
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

GAMES = 700
WORKERS = 24
SEED = 45
SIMULATIONS = 384
C_PUCT = 1.25
MAX_MOVES = 200
CONTEXTS = ("384:256", "1200:1200")
LANES = ("prospective_reused", "prospective_fresh384", "prospective_fresh_equiv")
A16_TARGET_CHECKPOINT = Path(
    "/tmp/azlite_fresh_p1_parent_adapter/artifacts/step_0016/checkpoint.npz"
)
A16_TARGET_CHECKPOINT_SHA = (
    "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34"
)
SUITE_SHA = "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04"
A16_SNAPSHOT_SHA = "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
INITIAL_OPTIMIZER_SHA = (
    "61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7"
)

_EVALUATOR: CheckpointEvaluator | None = None


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def worker_counts() -> list[int]:
    return [30] * 4 + [29] * 20


def worker_for(game_index: int) -> int:
    cursor = 0
    for worker_id, count in enumerate(worker_counts()):
        if cursor <= game_index < cursor + count:
            return worker_id
        cursor += count
    fail(f"game index outside generation contract: {game_index}")


def game_rng(game_index: int) -> random.Random:
    return random.Random(
        (SEED * 1_000_003) + game_index + (worker_for(game_index) * 9_973)
    )


def subtree_nodes(root: Node | None) -> int:
    return (
        0
        if root is None
        else 1 + sum(subtree_nodes(child) for child in root.children.values())
    )


def inherited_telemetry(root: Node | None) -> dict[str, int | float]:
    if root is None:
        return {
            "inherited_root_visit_count": 0,
            "inherited_child_visit_mass": 0,
            "inherited_visited_child_count": 0,
            "inherited_child_q_spread": 0.0,
            "retained_subtree_expanded_nodes": 0,
        }
    visits = np.asarray(
        [child.visit_count for child in root.children.values()], dtype=float
    )
    qs = np.asarray(
        [child.q_value for child in root.children.values() if child.visit_count],
        dtype=float,
    )
    return {
        "inherited_root_visit_count": int(root.visit_count),
        "inherited_child_visit_mass": int(visits.sum()),
        "inherited_visited_child_count": int(np.count_nonzero(visits)),
        "inherited_child_q_spread": float(qs.max() - qs.min()) if len(qs) else 0.0,
        "retained_subtree_expanded_nodes": subtree_nodes(root),
    }


def search(
    game: KalahGame,
    rng: random.Random,
    *,
    root: Node | None,
    reuse_subtree: bool,
    simulations: int,
    noisy: bool,
) -> tuple[list[float], Node, list[float]]:
    if _EVALUATOR is None:
        raise RuntimeError("worker evaluator is not initialized")
    engine = PUCT(
        evaluator=_EVALUATOR,
        simulations=simulations,
        c_puct=C_PUCT,
        rng=rng,
        root=root,
        fpu_mode="zero",
        reuse_subtree=reuse_subtree,
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
        fail("missing noisy root prior")
    policy = build_policy_target(
        visits,
        legal_moves=game.possible_moves(),
        temperature=1.0 if _PLY < 10 else 0.1,
        mode="default",
    )
    return policy, result_root, [float(value) for value in prior]


# The worker sets this immediately before each search; it preserves the PR #241
# visit-to-policy temperature schedule without passing mutable row state around.
_PLY = 0


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
        legal_moves = game.possible_moves()
        if not legal_moves:
            break
        _PLY = ply
        telemetry = inherited_telemetry(reusable_root)
        before = rng.getstate()
        reused_rng, fresh384_rng, fresh_equiv_rng = (
            random.Random(),
            random.Random(),
            random.Random(),
        )
        for clone in (reused_rng, fresh384_rng, fresh_equiv_rng):
            clone.setstate(before)
        noisy = ply < 10
        reused, reused_root, reused_prior = search(
            game,
            reused_rng,
            root=reusable_root,
            reuse_subtree=True,
            simulations=SIMULATIONS,
            noisy=noisy,
        )
        fresh384, _fresh_root, fresh_prior = search(
            game,
            fresh384_rng,
            root=None,
            reuse_subtree=False,
            simulations=SIMULATIONS,
            noisy=noisy,
        )
        equiv_simulations = SIMULATIONS + int(telemetry["inherited_child_visit_mass"])
        fresh_equiv, _equiv_root, equiv_prior = search(
            game,
            fresh_equiv_rng,
            root=None,
            reuse_subtree=False,
            simulations=equiv_simulations,
            noisy=noisy,
        )
        if not (
            np.array_equal(reused_prior, fresh_prior)
            and np.array_equal(reused_prior, equiv_prior)
        ):
            fail(f"root-noise mismatch game={game_index} ply={ply}")
        if ply == 0 and (
            telemetry["inherited_child_visit_mass"] != 0
            or not (reused == fresh384 == fresh_equiv)
        ):
            fail(f"ply-0 mismatch game={game_index}")
        state = encode_state(game.to_state(), input_encoding="kalah_v3")
        records.append(
            {
                "state": state,
                "player": game.current_player,
                "move_index": ply,
                "search_value": reused_root.q_value,
                "legal_moves": legal_moves,
                "policies": {
                    LANES[0]: reused,
                    LANES[1]: fresh384,
                    LANES[2]: fresh_equiv,
                },
                "telemetry": {
                    **telemetry,
                    "matched_fresh_equiv_simulations": equiv_simulations,
                    "gameplay_simulations": SIMULATIONS,
                    "target_simulations": equiv_simulations,
                    "total_simulations": SIMULATIONS + equiv_simulations,
                },
            }
        )
        move = sample_move(reused, legal_moves, reused_rng)
        reusable_root = reused_root.child_for_action(move)
        if not game.move(game.pit_index(move)):
            fail(f"illegal sampled move game={game_index} ply={ply}")
        rng = reused_rng
    if not game.over():
        fail(f"unterminated game={game_index}")
    winner = game.winner
    trajectory_hash = trajectory_hash_for_encoded_states(
        [record["state"] for record in records], winner=winner
    )
    profile = build_search_profile(
        kind="self_play",
        player_mode="puct",
        simulations=SIMULATIONS,
        c_puct=C_PUCT,
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
    telemetry_rows = []
    for record in records:
        common = {
            "state": record["state"],
            "value": derive_self_play_value_target(
                outcome_value=outcome_for_player(winner, record["player"]),
                search_value=record["search_value"],
                move_index=record["move_index"],
                mode="default",
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
            "simulations": SIMULATIONS,
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
        telemetry_rows.append(
            {
                "game_index": game_index,
                "move_index": record["move_index"],
                **record["telemetry"],
            }
        )
    return game_index, views, telemetry_rows


def generate(
    workdir: Path, workers: int
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(str(A16_TARGET_CHECKPOINT),),
    ) as executor:
        completed = list(executor.map(generate_game, range(GAMES), chunksize=1))
    completed.sort(key=lambda item: item[0])
    views = {
        lane: [
            row
            for _index, lane_rows, _telemetry in completed
            for row in lane_rows[lane]
        ]
        for lane in LANES
    }
    telemetry = [row for _index, _lane_rows, values in completed for row in values]
    for lane, rows in views.items():
        write_jsonl(workdir / "generated" / f"{lane}.jsonl", rows)
    write_jsonl(workdir / "generated" / "telemetry.jsonl", telemetry)
    return views, telemetry


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def non_policy_sha256(rows: list[dict[str, Any]]) -> str:
    payload = [
        {key: value for key, value in row.items() if key != "policy"} for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def target_diagnostics(
    rows: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
    targets: dict[str, list[list[float]]],
) -> dict[str, Any]:
    masses = np.asarray(
        [entry["inherited_child_visit_mass"] for entry in telemetry], dtype=float
    )
    edges = np.quantile(masses, [0.25, 0.5, 0.75])
    result = {}
    for left, right in (
        (LANES[0], LANES[1]),
        (LANES[0], LANES[2]),
        (LANES[1], LANES[2]),
    ):
        records = [
            isolation.policy_stats(a, b, row["legal_moves"])
            for row, a, b in zip(rows, targets[left], targets[right], strict=True)
        ]
        strata: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
        for record, telemetry_row in zip(records, telemetry, strict=True):
            strata[
                f"inherited_mass_quartile:{np.searchsorted(edges, telemetry_row['inherited_child_visit_mass'], side='right') + 1}"
            ].append(record)
        result[f"{left}_vs_{right}"] = {
            "overall": isolation.summarize(records),
            "by_inherited_mass_quartile": {
                key: isolation.summarize(value) for key, value in sorted(strata.items())
            },
        }
    closer = [
        isolation.policy_stats(reused, equiv, row["legal_moves"])["js"]
        < isolation.policy_stats(reused, fresh, row["legal_moves"])["js"]
        for row, reused, fresh, equiv in zip(
            rows, targets[LANES[0]], targets[LANES[1]], targets[LANES[2]], strict=True
        )
    ]
    result["fresh_equiv_js_closer_to_reused_fraction"] = float(np.mean(closer))
    return result


def train_lanes(
    rows: dict[str, list[dict[str, Any]]],
    pristine_model: dict[str, torch.Tensor],
    pristine_optimizer: dict[str, Any],
    parent: dict[str, torch.Tensor],
    workdir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    result, artifacts = {}, {}
    initial_sha = replay_train.optimizer_state_sha256(pristine_optimizer)
    targets = {name: [row["policy"] for row in values] for name, values in rows.items()}
    for lane in LANES:
        snapshots, optimizers, invocation = pr245.run_lane(
            rows[lane], pristine_model, pristine_optimizer, parent
        )
        lane_metrics = replay_train.metrics(
            rows[lane],
            snapshots,
            parent,
            pristine_model,
            copy.deepcopy(pristine_optimizer),
        )
        result[lane] = {
            "optimizer_invocation": invocation,
            "checkpoints": {},
            "metrics": lane_metrics,
            "cross_target_ce": {},
        }
        for step, state in snapshots.items():
            output = workdir / "train" / lane / f"step_{step:04d}"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": state, "optimizer": optimizers[step]},
                output.with_suffix(".pt"),
            )
            result[lane]["checkpoints"][str(step)] = pr245.checkpoint_record(
                state, optimizers[step]
            )
            result[lane]["cross_target_ce"][str(step)] = isolation.all_target_metrics(
                rows[lane], state, targets
            )
            if step == 16:
                artifacts[lane] = export(state, output, lane)
    repeated = pr245.repeated_lane_check(
        LANES[1], rows[LANES[1]], pristine_model, pristine_optimizer, parent
    )
    if (
        not repeated
        or replay_train.optimizer_state_sha256(pristine_optimizer) != initial_sha
    ):
        fail("optimizer isolation")
    result["optimizer_invariants"] = {
        "initial_sha256": initial_sha,
        "pristine_unchanged": replay_train.optimizer_state_sha256(pristine_optimizer)
        == initial_sha,
        "repeated_fresh384": repeated,
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
            (LANES[2], LANES[1]),
            (LANES[0], LANES[1]),
            (LANES[0], LANES[2]),
        )
    }
    return result


def classify(
    invariants: bool,
    diagnostics: dict[str, Any],
    training: dict[str, Any],
    arena: dict[str, Any],
) -> str:
    if not invariants:
        return "optimizer_or_generation_invariant_failure"
    fresh = arena[LANES[1]]["1200:1200"]["effect"]
    reused = arena[LANES[0]]["1200:1200"]["effect"]
    equiv = arena[LANES[2]]["1200:1200"]["effect"]
    fit = all(training[lane]["metrics"]["16"]["fit_fraction"] >= 0.25 for lane in LANES)
    close = abs(reused - equiv) <= abs(reused - fresh)
    if (
        fit
        and reused > fresh
        and equiv > fresh
        and close
        and diagnostics["fresh_equiv_js_closer_to_reused_fraction"] > 0.5
    ):
        return "prospective_effective_target_compute_replica"
    if reused > fresh and not close:
        return "reused_and_fresh_equiv_diverge_prospectively"
    if reused <= fresh and equiv <= fresh:
        return "effective_compute_effect_does_not_replicate"
    if equiv > fresh and reused <= fresh:
        return "fresh_equiv_improves_but_not_reused"
    return "inconclusive"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr246_prospective_effective_target_compute"),
    )
    parser.add_argument(
        "--canonical-suite",
        type=Path,
        default=Path("/tmp/azlite_opening_suite/medium_eval.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--skip-arena",
        action="store_true",
        help="Development-only: omit required ordinary-PUCT arenas.",
    )
    args = parser.parse_args()
    if args.workers != WORKERS:
        fail(f"workers must remain frozen at {WORKERS}")
    if (
        sha256_file(A16_TARGET_CHECKPOINT) != A16_TARGET_CHECKPOINT_SHA
        or sha256_file(args.canonical_suite) != SUITE_SHA
    ):
        fail("target evaluator or canonical suite hash mismatch")
    snapshot = torch.load(
        replay_train.A16_SNAPSHOT, map_location="cpu", weights_only=False
    )
    if (
        sha256_file(replay_train.A16_SNAPSHOT) != A16_SNAPSHOT_SHA
        or sha256_file(replay_train.P1_CHECKPOINT) != replay_train.P1_CHECKPOINT_SHA
    ):
        fail("A16 snapshot or P1 checkpoint hash mismatch")
    pristine_model, pristine_optimizer = replay_train.immutable_initial_state(snapshot)
    if replay_train.optimizer_state_sha256(pristine_optimizer) != INITIAL_OPTIMIZER_SHA:
        fail("initial optimizer hash mismatch")
    generated, all_telemetry = generate(args.workdir, args.workers)
    blocked, groups = replay_train.exclusion_hashes(args.canonical_suite)
    eligible, exclusions = {}, {}
    for lane in LANES:
        eligible[lane], exclusions[lane] = replay_train.filter_rows(
            generated[lane], blocked, groups
        )
    reference = eligible[LANES[0]]
    invariant_views = isolation.assert_views(reference, eligible)
    non_policy_hashes = {lane: non_policy_sha256(eligible[lane]) for lane in LANES}
    if len(set(non_policy_hashes.values())) != 1:
        fail("policy-excluded trajectory mismatch between replay views")
    eligible_keys = {(row["game_index"], row["move_index"]) for row in reference}
    telemetry = [
        row
        for row in all_telemetry
        if (row["game_index"], row["move_index"]) in eligible_keys
    ]
    if len(telemetry) != len(reference) or any(
        exclusions[lane] != exclusions[LANES[0]] for lane in LANES
    ):
        fail("inconsistent exclusions")
    targets = {lane: [row["policy"] for row in eligible[lane]] for lane in LANES}
    diagnostics = target_diagnostics(reference, telemetry, targets)
    compute = {
        "matched_fresh_equiv_simulations": quantiles(
            [entry["matched_fresh_equiv_simulations"] for entry in telemetry]
        ),
        "effective_multiplier": quantiles(
            [
                entry["matched_fresh_equiv_simulations"] / SIMULATIONS
                for entry in telemetry
            ]
        ),
        "fresh_equiv_total_compute": quantiles(
            [entry["total_simulations"] for entry in telemetry]
        ),
        "gameplay_simulations_per_move": SIMULATIONS,
        "target_simulations": "384 + inherited_child_visit_mass",
    }
    p1 = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1, replay_train.P1_CHECKPOINT)
    parent = {
        name: value.detach().cpu().clone() for name, value in p1.state_dict().items()
    }
    training, artifacts = train_lanes(
        eligible, pristine_model, pristine_optimizer, parent, args.workdir
    )
    generation_config = {
        "games": GAMES,
        "workers": WORKERS,
        "seed": SEED,
        "gameplay_simulations": SIMULATIONS,
        "c_puct": C_PUCT,
        "tree_reuse": True,
        "root_policy": "visit_count",
        "dirichlet": {"alpha": 0.3, "epsilon": 0.3, "plies": "0-9"},
        "input": "kalah_v3",
        "fpu_mode": "zero",
        "normalize_values": False,
        "policy_target_mode": "default",
        "value_target_mode": "default",
        "temperature": {"through_ply": 9, "early": 1.0, "late": 0.1},
        "start_state": "standard_4x6",
        "max_moves": MAX_MOVES,
    }
    result: dict[str, Any] = {
        "schema": "pr246_prospective_effective_target_compute_v1",
        "classification": "inconclusive",
        "frozen_artifacts": {
            "a16_snapshot_sha256": A16_SNAPSHOT_SHA,
            "initial_optimizer_sha256": INITIAL_OPTIMIZER_SHA,
            "p1_checkpoint_sha256": replay_train.P1_CHECKPOINT_SHA,
            "target_evaluator_sha256": A16_TARGET_CHECKPOINT_SHA,
            "canonical_suite_sha256": SUITE_SHA,
        },
        "generation": {
            **generation_config,
            "config_sha256": hashlib.sha256(
                json.dumps(
                    generation_config, sort_keys=True, separators=(",", ":")
                ).encode("ascii")
            ).hexdigest(),
            "replay_sha256": {
                lane: sha256_file(args.workdir / "generated" / f"{lane}.jsonl")
                for lane in LANES
            },
            "non_policy_state_outcome_sha256": non_policy_hashes,
            "telemetry_sha256": sha256_file(
                args.workdir / "generated" / "telemetry.jsonl"
            ),
        },
        "exclusions": exclusions[LANES[0]],
        "invariants": invariant_views,
        "target_diagnostics": diagnostics,
        "effective_compute": compute,
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
        result["classification"] = classify(
            True, diagnostics, training, result["canonical_arena"]
        )
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
