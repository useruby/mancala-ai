#!/usr/bin/env python3
# ruff: noqa: E402
"""Preregistered five-replay-seed reused-versus-fresh768 replication.

This runner intentionally has exactly two target lanes.  It freezes all ten
step-16 candidates before selecting S/T/U, and does not expose a budget or
per-seed selection option.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import random
import sys
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
from ml.alphazero_lite import run_pr252_phase_target_delta_attribution as pr252
from ml.alphazero_lite.evaluation_metrics import (
    paired_effect_difference,
    paired_opening_candidate_effect,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    export,
    new_model,
)
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

SEEDS = (48, 49, 50, 51, 52)
LANES = ("reused", "fresh768")
SUITE_SEEDS = {"S": 19042, "T": 20042, "U": 21042}
GAMES, WORKERS, BASE, MAX_MOVES = 700, 24, 384, 200
TARGET = Path("/tmp/azlite_fresh_p1_parent_adapter/artifacts/step_0016/checkpoint.npz")
A16_SHA = "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
ADAM_SHA = "61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7"
P1_SHA = "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9"
TARGET_SHA = "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34"
_EVALUATOR: CheckpointEvaluator | None = None


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


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


def search(
    game: KalahGame,
    rng: random.Random,
    *,
    root: Node | None,
    reuse: bool,
    simulations: int,
    noisy: bool,
    ply: int,
) -> tuple[list[float], Node, list[float]]:
    if _EVALUATOR is None:
        fail("worker evaluator not initialized")
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
    visits, result = engine.run(
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
            temperature=1.0 if ply < 10 else 0.1,
        ),
        result,
        [float(x) for x in prior],
    )


def generate_game(
    item: tuple[int, int],
) -> tuple[int, dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
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
        before = rng.getstate()
        runs = {}
        for lane, budget in (("reused", BASE), ("fresh768", 768)):
            clone = random.Random()
            clone.setstate(before)
            runs[lane] = (
                *search(
                    game,
                    clone,
                    root=root if lane == "reused" else None,
                    reuse=lane == "reused",
                    simulations=budget,
                    noisy=ply < 10,
                    ply=ply,
                ),
                clone,
            )
        if not np.array_equal(runs["reused"][2], runs["fresh768"][2]):
            fail(f"root-prior mismatch {seed}:{index}:{ply}")
        records.append(
            {
                "state": encode_state(game.to_state(), input_encoding="kalah_v3"),
                "player": game.current_player,
                "move_index": ply,
                "legal_moves": legal,
                "search_value": runs["reused"][1].q_value,
                "policies": {lane: runs[lane][0] for lane in LANES},
            }
        )
        # Only this clone may advance trajectory state or gameplay RNG.
        move = sample_move(runs["reused"][0], legal, runs["reused"][3])
        root, rng = runs["reused"][1].child_for_action(move), runs["reused"][3]
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
    for row in records:
        common = {
            "state": row["state"],
            "value": derive_self_play_value_target(
                outcome_value=outcome_for_player(winner, row["player"]),
                search_value=row["search_value"],
                move_index=row["move_index"],
            ),
            "player": row["player"],
            "move_index": row["move_index"],
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
            "simulations": BASE,
            "dirichlet_alpha": 0.3 if row["move_index"] < 10 else 0.0,
            "dirichlet_epsilon_for_sampling": 0.3 if row["move_index"] < 10 else 0.0,
            "dirichlet_epsilon_for_target": 0.3 if row["move_index"] < 10 else 0.0,
            "legal_moves": row["legal_moves"],
            "game_index": index,
            "game_completed": True,
            "game_length": len(records),
            "trajectory_hash": trajectory,
        }
        for lane in LANES:
            views[lane].append({**common, "policy": row["policies"][lane]})
        telemetry.append(
            {
                "game_index": index,
                "move_index": row["move_index"],
                "policies": row["policies"],
            }
        )
    return index, views, telemetry


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def non_policy_sha(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [{k: v for k, v in row.items() if k != "policy"} for row in rows],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def batch_plan_sha(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [batch.tolist() for batch in replay.batches(rows)],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def generate_seed(
    seed: int, output: Path
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=WORKERS, initializer=init_worker, initargs=(str(TARGET),)
    ) as executor:
        completed = list(
            executor.map(
                generate_game, ((seed, index) for index in range(GAMES)), chunksize=1
            )
        )
    completed.sort()
    views = {
        lane: [row for _, game, _ in completed for row in game[lane]] for lane in LANES
    }
    telemetry = [row for _, _, values in completed for row in values]
    for lane, rows in views.items():
        write_jsonl(output / "generated" / f"{lane}.jsonl", rows)
    write_jsonl(output / "generated" / "telemetry.jsonl", telemetry)
    return views, telemetry


def geometry(
    rows: list[dict[str, Any]], telemetry: list[dict[str, Any]]
) -> dict[str, Any]:
    stats = [
        isolation.policy_stats(
            a["policies"]["reused"], a["policies"]["fresh768"], row["legal_moves"]
        )
        for row, a in zip(rows, telemetry, strict=True)
    ]
    return isolation.summarize(stats)


def geometry_distribution(by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Summarize seed-level target geometry without using it for decisions."""
    result = {}
    for metric in (
        "mean_legal_policy_l1",
        "p50_legal_policy_l1",
        "p90_legal_policy_l1",
        "p99_legal_policy_l1",
        "mean_js",
        "top1_disagreement",
        "target_entropy_difference",
    ):
        values = np.asarray([by_seed[seed][metric] for seed in SEEDS], dtype=float)
        result[metric] = {
            "mean_of_seed_means": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "coefficient_of_variation": float(values.std(ddof=1) / values.mean())
            if values.mean() != 0
            else None,
        }
    return result


def train_seed(
    rows: dict[str, list[dict[str, Any]]],
    a16: dict[str, torch.Tensor],
    adam: dict[str, Any],
    parent: dict[str, torch.Tensor],
    output: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    initial, result, artifacts = replay.optimizer_state_sha256(adam), {}, {}
    targets = {
        lane: [row["policy"] for row in lane_rows] for lane, lane_rows in rows.items()
    }
    for lane in LANES:
        snapshots, optimizers, invocation = contract.run_lane(
            rows[lane], a16, adam, parent
        )
        metrics = replay.metrics(
            rows[lane], snapshots, parent, a16, copy.deepcopy(adam)
        )
        for step, state in snapshots.items():
            metrics[str(step)]["ce_all_target_families"] = isolation.all_target_metrics(
                rows[lane], state, targets
            )
            metrics[str(step)]["model_sha256"] = pr252.model_sha(state)
            metrics[str(step)]["optimizer_sha256"] = replay.optimizer_state_sha256(
                optimizers[step]
            )
            checkpoint = output / "train" / lane / f"step_{step:04d}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": state, "optimizer": optimizers[step]}, checkpoint)
        result[lane] = {"optimizer_invocation": invocation, "metrics": metrics}
        artifacts[lane] = export(
            snapshots[16], output / "train" / lane / "step_0016", f"pr256_{lane}"
        )
    repeated = contract.repeated_lane_check(
        "fresh768", rows["fresh768"], a16, adam, parent
    )
    # Reversing the two fixed lanes detects hidden state sharing.
    reordered = {
        lane: contract.run_lane(rows[lane], a16, adam, parent)[0]
        for lane in reversed(LANES)
    }
    order_ok = all(
        pr252.model_sha(reordered[lane][16])
        == result[lane]["metrics"]["16"]["model_sha256"]
        for lane in LANES
    )
    if not repeated or not order_ok or replay.optimizer_state_sha256(adam) != initial:
        fail("optimizer contamination, repeated identity, or lane-order dependence")
    result["optimizer_invariants"] = {
        "initial_sha256": initial,
        "pristine_unchanged": True,
        "repeated_fresh768": repeated,
        "lane_order_independent": order_ok,
    }
    return result, artifacts


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
    paths, manifest, prefixes = (
        {},
        {"consumed": registry_module.manifest(registry), "suites": {}},
        set(),
    )
    for label, seed in SUITE_SEEDS.items():
        selected = suites.select_diverse(
            [
                entry
                for entry in universe
                if suites.canonical_key(entry["state"]) not in used
                and not (pr251.prefix_keys([entry]) & (old_prefixes | prefixes))
            ],
            128,
            seed,
        )
        keys, current = pr249.suite_keys(selected), pr251.prefix_keys(selected)
        if len(keys) != 128 or keys & used or current & (old_prefixes | prefixes):
            fail(f"suite overlap {label}")
        path = output / "suites" / f"suite_{label}.jsonl"
        suites.write_suite_jsonl(selected, str(path))
        paths[label] = path
        manifest["suites"][label] = {
            "seed": seed,
            "sha256": sha256_file(path),
            "openings": 128,
            "consumed": True,
        }
        used |= keys
        prefixes |= current
    report = {
        "passed": True,
        "replay_seeds": list(SEEDS),
        "registry": registry_module.manifest(registry),
        "suites": manifest["suites"],
    }
    for label, path in paths.items():
        selected = suites.load_suite_jsonl(str(path))
        keys, prefixes = pr249.suite_keys(selected), pr251.prefix_keys(selected)
        report[label] = {
            "final_overlap_registry": len(keys & registry_module.final_keys(registry)),
            "prefix_overlap_registry": len(
                prefixes & registry_module.prefix_keys(registry)
            ),
            "replay_state_overlap": sum(
                tuple(encode_state(row["state"], input_encoding="kalah_v3"))
                in replay_states
                for row in selected
            ),
        }
        if any(report[label].values()):
            fail(f"suite preflight overlap {label}")
    return paths, manifest, report


def evaluate(
    candidates: dict[int, dict[str, Path]],
    suites_paths: dict[str, Path],
    output: Path,
    context: str,
    frozen_manifest: Path,
) -> dict[str, Any]:
    if not frozen_manifest.is_file():
        fail("arena attempted before frozen manifest")
    if not json.loads(frozen_manifest.read_text(encoding="utf-8"))["preflight"][
        "passed"
    ]:
        fail("arena attempted before successful preflight")
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
        for seed in SEEDS:
            result[label][str(seed)] = {}
            for lane in LANES:
                records = isolation.arena_records(
                    output / "arena" / context / label,
                    candidates[seed][lane],
                    replay.P1_CHECKPOINT.parent / "artifact",
                    context,
                    f"seed{seed}_{lane}",
                    WORKERS,
                    suite,
                )
                result[label][str(seed)][lane] = {
                    "effect": paired_opening_candidate_effect(records, control),
                    "wdl": pr244.win_draw_loss(records),
                }
    return result


def analyze(evaluation: dict[str, Any]) -> dict[str, Any]:
    per_seed, rng = {}, np.random.default_rng(42)
    for seed in SEEDS:
        suite_values, suite_effects = [], {}
        absolute = {lane: [] for lane in LANES}
        for label in SUITE_SEEDS:
            item = evaluation[label][str(seed)]
            diff = paired_effect_difference(
                item["fresh768"]["effect"], item["reused"]["effect"]
            )
            values = np.asarray(list(diff["per_opening_effect"].values()), dtype=float)
            suite_values.append(values)
            suite_effects[label] = float(values.mean())
            for lane in LANES:
                absolute[lane].append(
                    float(item[lane]["effect"]["paired_candidate_effect"])
                )
        draws = np.asarray(
            [
                rng.choice(suite_values[rng.integers(0, 3)], 128, replace=True).mean()
                for _ in range(10_000)
            ]
        )
        per_seed[str(seed)] = {
            "delta": float(np.mean(suite_values)),
            "opening_bootstrap_ci": {
                "lower_95": float(np.quantile(draws, 0.025)),
                "upper_95": float(np.quantile(draws, 0.975)),
            },
            "suite_values": suite_effects,
            "sign_consistency": sum(value > 0 for value in suite_effects.values()),
            "fresh768_absolute_p1_effect": float(np.mean(absolute["fresh768"])),
            "reused_absolute_p1_effect": float(np.mean(absolute["reused"])),
        }
    deltas = np.asarray([per_seed[str(seed)]["delta"] for seed in SEEDS])
    draws = []
    for _ in range(10_000):
        sampled = []
        for seed in rng.choice(SEEDS, len(SEEDS), replace=True):
            label = list(SUITE_SEEDS)[rng.integers(0, 3)]
            diff = paired_effect_difference(
                evaluation[label][str(seed)]["fresh768"]["effect"],
                evaluation[label][str(seed)]["reused"]["effect"],
            )
            values = np.asarray(list(diff["per_opening_effect"].values()), dtype=float)
            sampled.append(rng.choice(values, len(values), replace=True).mean())
        draws.append(float(np.mean(sampled)))
    return {
        "per_replay_seed": per_seed,
        "primary": {
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "sd": float(deltas.std(ddof=1)),
            "min": float(deltas.min()),
            "max": float(deltas.max()),
            "positive_seeds": int((deltas > 0).sum()),
            "hierarchical_replay_suite_opening_ci": {
                "lower_95": float(np.quantile(draws, 0.025)),
                "upper_95": float(np.quantile(draws, 0.975)),
            },
        },
    }


def variance_analysis(
    primary: dict[str, Any], training: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    result = {}
    for name, values in {
        "absolute_strength": {
            lane: [
                primary["per_replay_seed"][str(seed)][f"{lane}_absolute_p1_effect"]
                for seed in SEEDS
            ]
            for lane in LANES
        },
        "adapter_norm": {
            lane: [
                training[seed][lane]["metrics"]["16"]["adapter_norm"] for seed in SEEDS
            ]
            for lane in LANES
        },
        "policy_distance_from_p1": {
            lane: [
                training[seed][lane]["metrics"]["16"]["legal_policy_l1_vs_p1"]
                for seed in SEEDS
            ]
            for lane in LANES
        },
    }.items():
        result[name] = {
            lane: float(np.var(value, ddof=1)) for lane, value in values.items()
        }
    return result


def classify(
    analysis: dict[str, Any],
    training: dict[int, dict[str, Any]],
    variance: dict[str, Any],
    shallow: dict[str, Any] | None = None,
) -> str:
    primary, deltas = (
        analysis["primary"],
        [analysis["per_replay_seed"][str(seed)]["delta"] for seed in SEEDS],
    )
    fits = all(
        training[seed][lane]["metrics"]["16"]["fit_fraction"] >= 0.25
        for seed in SEEDS
        for lane in LANES
    )
    success = (
        primary["mean_delta"] > 0
        and primary["hierarchical_replay_suite_opening_ci"]["lower_95"] > 0
        and primary["positive_seeds"] >= 4
        and min(deltas) >= -0.02
        and fits
    )
    if success and shallow is not None:
        values = [shallow["per_replay_seed"][str(seed)]["delta"] for seed in SEEDS]
        if (
            shallow["primary"]["mean_delta"] < -0.02
            or sum(value < -0.02 for value in values) >= 2
        ):
            return "fresh768_high_budget_gain_with_shallow_harm"
    if success:
        return "fresh768_robust_across_replay_seeds"
    variance_reduced = (
        variance["absolute_strength"]["fresh768"]
        < variance["absolute_strength"]["reused"]
    )
    if primary["mean_delta"] > 0:
        return "fresh768_improves_mean_but_is_seed_sensitive"
    return (
        "fresh768_reduces_variance_without_mean_gain"
        if variance_reduced
        else "fresh768_not_robust"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr256_fresh768_replay_replication"),
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--run-shallow",
        action="store_true",
        help="only after retained primary results are frozen",
    )
    args = parser.parse_args()
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
    registry = registry_module.load(args.workdir)
    blocked, groups = replay.exclusion_hashes(pr249.CANONICAL_SUITE)
    consumed = set().union(
        *(replay.canonical_arena_hashes(spec.path) for spec in registry.values())
    )
    blocked |= consumed
    groups = {**groups, "consumed_evaluation_openings": consumed}
    eligible, training, candidates, replays, geometry_by_seed, replay_manifest = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    for seed in SEEDS:
        seed_dir = args.workdir / f"seed{seed}"
        generated, telemetry = generate_seed(seed, seed_dir)
        views, exclusions = {}, {}
        for lane in LANES:
            views[lane], exclusions[lane] = replay.filter_rows(
                generated[lane], blocked, groups
            )
        hashes = {lane: non_policy_sha(rows) for lane, rows in views.items()}
        batches = {lane: batch_plan_sha(rows) for lane, rows in views.items()}
        if (
            len(set(hashes.values())) != 1
            or len(set(batches.values())) != 1
            or exclusions["reused"] != exclusions["fresh768"]
        ):
            fail(f"replay view mismatch seed={seed}")
        isolation.assert_views(views["reused"], views)
        keys = {(row["game_index"], row["move_index"]) for row in views["reused"]}
        filtered_telemetry = [
            row for row in telemetry if (row["game_index"], row["move_index"]) in keys
        ]
        if len(filtered_telemetry) != len(views["reused"]):
            fail(f"telemetry mismatch seed={seed}")
        p1 = new_model(torch.device("cpu"))
        load_checkpoint_into_model(p1, replay.P1_CHECKPOINT)
        parent = {
            key: value.detach().cpu().clone() for key, value in p1.state_dict().items()
        }
        training[seed], candidates[seed] = train_seed(
            views, a16, adam, parent, seed_dir
        )
        eligible[seed], replays[seed], geometry_by_seed[seed] = (
            views,
            views["reused"],
            geometry(views["reused"], filtered_telemetry),
        )
        replay_manifest[str(seed)] = {
            "policy_excluded_state_outcome_sha256": hashes["reused"],
            "replay_sha256": {
                lane: sha256_file(seed_dir / "generated" / f"{lane}.jsonl")
                for lane in LANES
            },
            "eligible_rows": len(views["reused"]),
            "batch_plan_sha256": batches["reused"],
            "exclusions": exclusions["reused"],
        }
    paths, suite_manifest, preflight = seal_suites(args.workdir, registry, replays)
    frozen = {
        "schema": "alphazero_lite_pr256_fresh768_replay_replication_v1",
        "ordering": [
            "all_replays",
            "all_candidates",
            "registry",
            "STU",
            "preflight",
            "manifest",
            "arena",
        ],
        "artifacts": {
            "a16": A16_SHA,
            "adam": ADAM_SHA,
            "p1": P1_SHA,
            "evaluator": TARGET_SHA,
        },
        "replays": replay_manifest,
        "candidates": training,
        "suite_manifest": suite_manifest,
        "preflight": preflight,
    }
    registry_path = args.workdir / "suite_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "consumed": registry_module.manifest(registry),
                "newly_consumed": suite_manifest["suites"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preflight_path = args.workdir / "preflight_audit.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen["suite_registry_sha256"] = sha256_file(registry_path)
    frozen["preflight_sha256"] = sha256_file(preflight_path)
    frozen_path = args.workdir / "frozen_manifest.json"
    frozen_path.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen_manifest_sha256 = sha256_file(frozen_path)
    if args.freeze_only:
        return
    primary_eval = evaluate(candidates, paths, args.workdir, "1200:1200", frozen_path)
    primary = analyze(primary_eval)
    primary_path = args.workdir / "primary_1200_results.json"
    primary_path.write_text(
        json.dumps(primary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    primary_sha256 = sha256_file(primary_path)
    shallow = (
        analyze(evaluate(candidates, paths, args.workdir, "384:256", frozen_path))
        if args.run_shallow
        else None
    )
    variance = variance_analysis(primary, training)
    total_states = sum(
        len(read_jsonl(args.workdir / f"seed{seed}" / "generated" / "reused.jsonl"))
        for seed in SEEDS
    )
    result = {
        **frozen,
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "target_diagnostics": geometry_by_seed,
        "target_diagnostic_distribution": geometry_distribution(geometry_by_seed),
        "primary_evaluation": primary,
        "primary_1200_results_sha256": primary_sha256,
        "shallow_evaluation": shallow,
        "variance_analysis": variance,
        "compute": {
            "reused_simulations_per_move": 384,
            "fresh768_simulations_per_move": 1152,
            "fresh768_multiplier": 3.0,
            "five_seed_authoritative_states": total_states,
            "five_seed_total_search_simulations": 1152 * total_states,
        },
        "classification": classify(primary, training, variance, shallow),
    }
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
