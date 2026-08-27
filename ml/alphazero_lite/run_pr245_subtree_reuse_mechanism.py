#!/usr/bin/env python3
# ruff: noqa: E402
"""Isolate inherited subtree statistics from effective search mass in PR #245.

This runner never generates a trajectory.  It replays each committed ordinary
game sequentially, verifies its sampled actions and states, and changes only
the policy field in two matched-RNG counterfactual replay views.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
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
    encode_state,
    sample_move,
    standard_start_state,
)
from ml.alphazero_lite.train import load_checkpoint_into_model

SOURCE_SHA = "6671e248af4a4c82e1155c798cb7490cd66cd80dc10b203c97d89dced94527f2"
EXPECTED_ELIGIBLE_ROWS = 27350
SUITE_SHA = "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04"
A16_SNAPSHOT_SHA = "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
INITIAL_OPTIMIZER_SHA = (
    "61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7"
)
TARGET_CHECKPOINT = Path(
    "/tmp/azlite_fresh_p1_parent_adapter/artifacts/step_0016/checkpoint.npz"
)
TARGET_CHECKPOINT_SHA = (
    "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34"
)
GENERATED_GAMES = 700
GENERATED_WORKERS = 24
GENERATED_SEED = 44
CONTEXTS = ("384:256", "1200:1200")
LANES = ("original_reused", "fresh_matched_384", "fresh_matched_total")

_WORKER_EVALUATOR: CheckpointEvaluator | None = None


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def generation_worker_for(game_index: int) -> int:
    """Recover PR #241's 700-game/24-worker partition from its provenance."""
    counts = [30] * 4 + [29] * 20
    cursor = 0
    for worker_id, count in enumerate(counts):
        if cursor <= game_index < cursor + count:
            return worker_id
        cursor += count
    fail(f"game index outside frozen generation contract: {game_index}")


def gameplay_rng(game_index: int) -> random.Random:
    worker_id = generation_worker_for(game_index)
    return random.Random(
        (GENERATED_SEED * 1_000_003) + game_index + (worker_id * 9_973)
    )


def entropy(values: np.ndarray) -> float:
    positive = values > 0.0
    return float(-np.sum(values[positive] * np.log(values[positive])))


def subtree_node_count(root: Node | None) -> int:
    if root is None:
        return 0
    return 1 + sum(subtree_node_count(child) for child in root.children.values())


def inherited_telemetry(
    root: Node | None, base_simulations: int
) -> dict[str, int | float]:
    if root is None:
        return {
            "inherited_root_visit_count": 0,
            "inherited_child_visit_mass": 0,
            "inherited_visited_child_count": 0,
            "inherited_child_q_spread": 0.0,
            "inherited_visit_entropy": 0.0,
            "inherited_visit_concentration": 0.0,
            "retained_subtree_expanded_nodes": 0,
            "effective_multiplier": 1.0,
        }
    visits = np.asarray(
        [child.visit_count for child in root.children.values()], dtype=float
    )
    visited = visits > 0
    mass = int(visits.sum())
    qs = np.asarray(
        [child.q_value for child in root.children.values() if child.visit_count],
        dtype=float,
    )
    distribution = (
        visits[visited] / visits[visited].sum() if np.any(visited) else visits
    )
    return {
        "inherited_root_visit_count": int(root.visit_count),
        "inherited_child_visit_mass": mass,
        "inherited_visited_child_count": int(visited.sum()),
        "inherited_child_q_spread": float(qs.max() - qs.min()) if len(qs) else 0.0,
        "inherited_visit_entropy": entropy(distribution) if len(distribution) else 0.0,
        "inherited_visit_concentration": float(distribution.max())
        if len(distribution)
        else 0.0,
        "retained_subtree_expanded_nodes": subtree_node_count(root),
        "effective_multiplier": (base_simulations + mass) / base_simulations,
    }


def search(
    evaluator: CheckpointEvaluator,
    game: KalahGame,
    row: dict[str, Any],
    rng: random.Random,
    *,
    root: Node | None,
    reuse_subtree: bool,
    simulations: int,
) -> tuple[list[float], Node, list[float]]:
    options = row["teacher_search_profile"]["search_options"]
    noise_enabled = bool(row["action_sampling_noise_enabled"])
    engine = PUCT(
        evaluator=evaluator,
        simulations=int(simulations),
        c_puct=float(row["teacher_search_profile"]["c_puct"]),
        rng=rng,
        root=root,
        fpu_mode=str(options["fpu_mode"]),
        reuse_subtree=reuse_subtree,
        normalize_values=bool(options["normalize_values"]),
        root_policy_mode=str(options["root_policy_mode"]),
        tactical_root_bias=float(options["tactical_root_bias"]),
        root_temperature=float(options.get("root_temperature", 0.0)),
    )
    visits, result_root = engine.run(
        game,
        dirichlet_alpha=float(row["dirichlet_alpha"]) if noise_enabled else None,
        dirichlet_epsilon=float(row["target_dirichlet_epsilon"])
        if noise_enabled
        else 0.0,
    )
    prior = engine.root_summary()["root_prior_telemetry"]["after"]
    if prior is None:
        fail("missing root-prior telemetry")
    assert isinstance(prior, list)
    policy = build_policy_target(
        visits,
        legal_moves=[int(move) for move in row["legal_moves"]],
        temperature=1.0 if int(row["move_index"]) < 10 else 0.1,
        mode=str(row["policy_target_mode"]),
    )
    return policy, result_root, [float(value) for value in prior]


def init_worker(checkpoint: str) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = CheckpointEvaluator(Path(checkpoint), input_encoding="kalah_v3")


def reconstruct_game(
    item: tuple[int, list[dict[str, Any]]],
) -> tuple[int, list[dict[str, Any]]]:
    """Reconstruct one committed game and its two policy-only counterfactuals."""
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("target evaluator is not initialized")
    game_index, rows = item
    rows = sorted(rows, key=lambda row: int(row["move_index"]))
    if not rows or [int(row["move_index"]) for row in rows] != list(range(len(rows))):
        fail(f"non-contiguous plies in game {game_index}")
    game, rng, reusable_root = (
        KalahGame.from_state(standard_start_state()),
        gameplay_rng(game_index),
        None,
    )
    records: list[dict[str, Any]] = []
    for ply, row in enumerate(rows):
        encoded = encode_state(game.to_state(), input_encoding="kalah_v3")
        if encoded != row["state"]:
            fail(f"state mismatch game={game_index} ply={ply}")
        base = int(row["simulations"])
        telemetry = inherited_telemetry(reusable_root, base)
        before_search = rng.getstate()
        warm_rng, fresh384_rng, fresh_total_rng = (
            random.Random(),
            random.Random(),
            random.Random(),
        )
        warm_rng.setstate(before_search)
        fresh384_rng.setstate(before_search)
        fresh_total_rng.setstate(before_search)
        reused, warm_root, warm_prior = search(
            _WORKER_EVALUATOR,
            game,
            row,
            warm_rng,
            root=reusable_root,
            reuse_subtree=True,
            simulations=base,
        )
        fresh384, _fresh_root, fresh_prior = search(
            _WORKER_EVALUATOR,
            game,
            row,
            fresh384_rng,
            root=None,
            reuse_subtree=False,
            simulations=base,
        )
        total_simulations = base + int(telemetry["inherited_child_visit_mass"])
        fresh_total, _total_root, total_prior = search(
            _WORKER_EVALUATOR,
            game,
            row,
            fresh_total_rng,
            root=None,
            reuse_subtree=False,
            simulations=total_simulations,
        )
        if not (
            np.array_equal(warm_prior, fresh_prior)
            and np.array_equal(warm_prior, total_prior)
        ):
            fail(f"root-noise mismatch game={game_index} ply={ply}")
        if not np.array_equal(
            np.asarray(reused, dtype=np.float32),
            np.asarray(row["policy"], dtype=np.float32),
        ):
            fail(f"reconstructed target mismatch game={game_index} ply={ply}")
        if ply == 0 and (
            telemetry["inherited_child_visit_mass"] != 0
            or not (reused == fresh384 == fresh_total)
        ):
            fail(f"ply-0 matched-RNG mismatch game={game_index}")
        move = sample_move(reused, [int(move) for move in row["legal_moves"]], warm_rng)
        if not game.move(game.pit_index(move)):
            fail(f"sampled illegal move game={game_index} ply={ply}")
        if (
            ply + 1 < len(rows)
            and encode_state(game.to_state(), input_encoding="kalah_v3")
            != rows[ply + 1]["state"]
        ):
            fail(f"sampled move/state mismatch game={game_index} ply={ply}")
        records.append(
            {
                "game_index": game_index,
                "move_index": ply,
                "fresh_matched_384": fresh384,
                "fresh_matched_total": fresh_total,
                "matched_simulations": total_simulations,
                **telemetry,
            }
        )
        rng = warm_rng
        reusable_root = warm_root.child_for_action(move)
    if (
        not game.over()
        or game.winner != rows[0]["winner"]
        or len(rows) != int(rows[0]["game_length"])
    ):
        fail(f"terminal outcome mismatch game={game_index}")
    return game_index, records


def reconstruct(
    rows: list[dict[str, Any]], checkpoint: Path, workers: int
) -> dict[tuple[int, int], dict[str, Any]]:
    games: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        games[int(row["game_index"])].append(row)
    if sorted(games) != list(range(GENERATED_GAMES)):
        fail("frozen replay does not contain exactly the 700 original games")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(str(checkpoint),)
    ) as executor:
        completed = executor.map(reconstruct_game, sorted(games.items()), chunksize=1)
        records = [record for _index, values in completed for record in values]
    return {
        (int(record["game_index"]), int(record["move_index"])): record
        for record in records
    }


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        key: float(np.percentile(array, percentile))
        for key, percentile in (("p50", 50), ("p90", 90), ("p99", 99))
    }


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        ties = np.flatnonzero(values == value)
        result[ties] = result[ties].mean()
    return result


def target_diagnostics(
    rows: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
    targets: dict[str, list[list[float]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    mass = np.asarray(
        [entry["inherited_child_visit_mass"] for entry in telemetry], dtype=float
    )
    multiplier = np.asarray(
        [entry["effective_multiplier"] for entry in telemetry], dtype=float
    )
    mass_edges, multiplier_edges = (
        np.quantile(mass, [0.25, 0.5, 0.75]),
        np.quantile(multiplier, [0.25, 0.5, 0.75]),
    )
    for left, right in (
        ("original_reused", "fresh_matched_384"),
        ("original_reused", "fresh_matched_total"),
        ("fresh_matched_384", "fresh_matched_total"),
    ):
        records = [
            isolation.policy_stats(a, b, row["legal_moves"])
            for row, a, b in zip(rows, targets[left], targets[right], strict=True)
        ]
        groups: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
        for row, item, record in zip(rows, telemetry, records, strict=True):
            groups[
                f"inherited_mass_quartile:{np.searchsorted(mass_edges, item['inherited_child_visit_mass'], side='right') + 1}"
            ].append(record)
            groups[
                f"effective_multiplier_quartile:{np.searchsorted(multiplier_edges, item['effective_multiplier'], side='right') + 1}"
            ].append(record)
            groups[f"phase:{isolation.phase(row)}"].append(record)
            groups[f"legal_move_count:{len(row['legal_moves'])}"].append(record)
            groups[f"root_noise:{bool(row['action_sampling_noise_enabled'])}"].append(
                record
            )
        summary = isolation.summarize(records)
        if left == "original_reused":
            summary["spearman_inherited_mass_vs_js"] = float(
                np.corrcoef(rank(mass), rank(np.asarray([r["js"] for r in records])))[
                    0, 1
                ]
            )
        result[f"{left}_vs_{right}"] = {
            "overall": summary,
            "by_stratum": {
                key: isolation.summarize(value) for key, value in sorted(groups.items())
            },
        }
    return result


def effective_telemetry(
    telemetry: list[dict[str, Any]],
    targets: dict[str, list[list[float]]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    multipliers = [float(entry["effective_multiplier"]) for entry in telemetry]
    toward_top1, closer_js = [], []
    for row, original, fresh, total in zip(
        rows,
        targets["original_reused"],
        targets["fresh_matched_384"],
        targets["fresh_matched_total"],
        strict=True,
    ):
        legal = row["legal_moves"]
        top_original = legal[int(np.argmax(np.asarray(original)[legal]))]
        toward_top1.append(
            int(
                legal[int(np.argmax(np.asarray(total)[legal]))] == top_original
                and legal[int(np.argmax(np.asarray(fresh)[legal]))] != top_original
            )
        )
        closer_js.append(
            int(
                isolation.policy_stats(original, total, legal)["js"]
                < isolation.policy_stats(original, fresh, legal)["js"]
            )
        )
    return {
        "effective_multiplier": {
            "mean": float(np.mean(multipliers)),
            "max": float(np.max(multipliers)),
            **percentile_summary(multipliers),
        },
        "fresh_total_top1_toward_original_rate": float(np.mean(toward_top1)),
        "fresh_total_js_closer_rate": float(np.mean(closer_js)),
        "matched_simulations": {
            "mean": float(
                np.mean([entry["matched_simulations"] for entry in telemetry])
            ),
            "max": int(max(entry["matched_simulations"] for entry in telemetry)),
            **percentile_summary([entry["matched_simulations"] for entry in telemetry]),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def train_lanes(
    rows: dict[str, list[dict[str, Any]]],
    pristine_model: dict[str, torch.Tensor],
    pristine_optimizer: dict[str, Any],
    parent: dict[str, torch.Tensor],
    workdir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    result, artifacts = {}, {}
    initial_fingerprint = replay_train.optimizer_state_sha256(pristine_optimizer)
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
            "checkpoints": {
                str(step): pr245.checkpoint_record(snapshots[step], optimizers[step])
                for step in replay_train.STEPS
            },
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
            result[lane]["cross_target_ce"][str(step)] = isolation.all_target_metrics(
                rows[lane],
                state,
                {name: [row["policy"] for row in view] for name, view in rows.items()},
            )
            if step == 16:
                artifacts[lane] = export(state, output, f"pr245_subtree_{lane}")
    if replay_train.optimizer_state_sha256(pristine_optimizer) != initial_fingerprint:
        fail("pristine optimizer mutation")
    repeated = pr245.repeated_lane_check(
        "fresh_matched_384",
        rows["fresh_matched_384"],
        pristine_model,
        pristine_optimizer,
        parent,
    )
    if not repeated:
        fail("repeated lane nonidentity")
    result["optimizer_invariants"] = {
        "initial_sha256": initial_fingerprint,
        "pristine_unchanged": replay_train.optimizer_state_sha256(pristine_optimizer)
        == initial_fingerprint,
        "repeated_fresh_matched_384": repeated,
    }
    return result, artifacts


def positive_control(workdir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    """Require the original view to match the clean PR #245 first lane exactly."""
    historical_root = Path("/tmp/azlite_onpolicy_shadow_replay")
    historical_metrics = json.loads(
        (historical_root / "training_summary.json").read_text(encoding="utf-8")
    )["lanes"]["ordinary_onpolicy"]["metrics"]
    result: dict[str, Any] = {}
    for step in replay_train.STEPS:
        actual = torch.load(
            workdir / "train" / "original_reused" / f"step_{step:04d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        expected = torch.load(
            historical_root / "ordinary_onpolicy_train" / f"step_{step:04d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        checkpoint = {
            "model_sha_match": pr245.state_sha256(actual["model"])
            == pr245.state_sha256(expected["model"]),
            "optimizer_sha_match": replay_train.optimizer_state_sha256(
                actual["optimizer"]
            )
            == replay_train.optimizer_state_sha256(expected["optimizer"]),
            "adapter_tensor_sha_match": pr245.adapter_sha256(actual["model"])
            == pr245.adapter_sha256(expected["model"]),
        }
        ce_match = all(
            np.isclose(
                metrics["original_reused"]["metrics"][str(step)][key],
                historical_metrics[str(step)][key],
                atol=0.0,
                rtol=0.0,
            )
            for key in ("ce_search", "ce_p1", "ce_beta095")
        )
        result[str(step)] = {**checkpoint, "ce_metrics_match": ce_match}
    if not all(all(check.values()) for check in result.values()):
        fail("original_reused did not reproduce clean PR #245 positive control")
    return result


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
            ("original_reused", "fresh_matched_384"),
            ("fresh_matched_total", "fresh_matched_384"),
            ("original_reused", "fresh_matched_total"),
        )
    }
    return result


def classify(arena: dict[str, Any]) -> str:
    original, fresh, total = (arena[lane]["1200:1200"]["effect"] for lane in LANES)
    if original != 0.041015625:
        return "reused_target_gain_not_reproducible"
    if fresh >= 0.02:
        return "matched_rng_fresh_target_recovers_gain"
    if total >= 0.02 and abs(total - original) <= abs(fresh - original):
        return "extra_effective_search_budget_explains_reused_gain"
    if total < 0.0:
        return "path_dependent_subtree_statistics_explain_gain"
    return "target_compute_increases_fit_but_not_strength"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr245_subtree_reuse_mechanism"),
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("/tmp/azlite_onpolicy_shadow_replay/ordinary_onpolicy.jsonl"),
    )
    parser.add_argument(
        "--canonical-suite",
        type=Path,
        default=Path("/tmp/azlite_opening_suite/medium_eval.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--skip-arena",
        action="store_true",
        help="Development-only: omit the required ordinary-PUCT arena.",
    )
    args = parser.parse_args()
    if (
        sha256_file(args.replay) != SOURCE_SHA
        or sha256_file(args.canonical_suite) != SUITE_SHA
        or sha256_file(TARGET_CHECKPOINT) != TARGET_CHECKPOINT_SHA
    ):
        fail("frozen replay, suite, or target evaluator hash mismatch")
    source = read_jsonl(args.replay)
    reconstructed = reconstruct(source, TARGET_CHECKPOINT, args.workers)
    all_telemetry = [
        reconstructed[(int(row["game_index"]), int(row["move_index"]))]
        for row in source
    ]
    write_jsonl(args.workdir / "telemetry.jsonl", all_telemetry)
    blocked, groups = replay_train.exclusion_hashes(args.canonical_suite)
    eligible, exclusions = replay_train.filter_rows(source, blocked, groups)
    if len(eligible) != EXPECTED_ELIGIBLE_ROWS:
        fail("eligible row count mismatch")
    views = {"original_reused": [copy.deepcopy(row) for row in eligible]}
    for lane in LANES[1:]:
        views[lane] = [
            {
                **copy.deepcopy(row),
                "policy": reconstructed[
                    (int(row["game_index"]), int(row["move_index"]))
                ][lane],
            }
            for row in eligible
        ]
    invariants = isolation.assert_views(eligible, views)
    for lane, rows in views.items():
        write_jsonl(args.workdir / "derived" / f"{lane}.jsonl", rows)
    telemetry = [
        reconstructed[(int(row["game_index"]), int(row["move_index"]))]
        for row in eligible
    ]
    targets = {lane: [row["policy"] for row in rows] for lane, rows in views.items()}
    snapshot = torch.load(
        replay_train.A16_SNAPSHOT, map_location="cpu", weights_only=False
    )
    if (
        sha256_file(replay_train.A16_SNAPSHOT) != A16_SNAPSHOT_SHA
        or sha256_file(replay_train.P1_CHECKPOINT) != replay_train.P1_CHECKPOINT_SHA
    ):
        fail("A16 or P1 artifact hash mismatch")
    pristine_model, pristine_optimizer = replay_train.immutable_initial_state(snapshot)
    if replay_train.optimizer_state_sha256(pristine_optimizer) != INITIAL_OPTIMIZER_SHA:
        fail("initial optimizer hash mismatch")
    p1 = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1, replay_train.P1_CHECKPOINT)
    parent = {
        name: value.detach().cpu().clone() for name, value in p1.state_dict().items()
    }
    training, artifacts = train_lanes(
        views, pristine_model, pristine_optimizer, parent, args.workdir
    )
    reproduction = positive_control(args.workdir, training)
    result: dict[str, Any] = {
        "schema": "pr245_subtree_reuse_mechanism_v1",
        "classification": "inconclusive",
        "frozen_artifacts": {
            "ordinary_replay_sha256": SOURCE_SHA,
            "eligible_rows": EXPECTED_ELIGIBLE_ROWS,
            "a16_snapshot_sha256": A16_SNAPSHOT_SHA,
            "p1_checkpoint_sha256": replay_train.P1_CHECKPOINT_SHA,
            "canonical_suite_sha256": SUITE_SHA,
            "initial_optimizer_sha256": INITIAL_OPTIMIZER_SHA,
        },
        "exclusions": exclusions,
        "derived_replay_sha256": {
            lane: sha256_file(args.workdir / "derived" / f"{lane}.jsonl")
            for lane in LANES
        },
        "telemetry_sha256": sha256_file(args.workdir / "telemetry.jsonl"),
        "invariants": {**invariants, "original_pr245_positive_control": reproduction},
        "target_diagnostics": target_diagnostics(eligible, telemetry, targets),
        "effective_search_telemetry": effective_telemetry(telemetry, targets, eligible),
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
        original_arena = result["canonical_arena"]["original_reused"]
        if (
            original_arena["384:256"]["effect"] != -0.01953125
            or original_arena["1200:1200"]["effect"] != 0.041015625
        ):
            fail("original_reused arena did not reproduce clean PR #245")
        result["classification"] = classify(result["canonical_arena"])
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
