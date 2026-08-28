#!/usr/bin/env python3
# ruff: noqa: E402
"""Replay PR #246 with fixed policy-target search budgets for PR #247.

The committed reused trajectory is the sole replay source.  It supplies action
succession and inherited-tree geometry, but is never a training lane.
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
from ml.alphazero_lite import (
    run_pr241_optimizer_isolation_reproduction as train_mechanism,
)
from ml.alphazero_lite import run_pr241_policy_target_noise_isolation as isolation
from ml.alphazero_lite import run_pr242_target_entropy_factorization as pr244
from ml.alphazero_lite import run_pr245_subtree_reuse_mechanism as pr245
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

SOURCE_ROOT = Path("/tmp/azlite_pr246_prospective_effective_target_compute")
SOURCE = SOURCE_ROOT / "generated/prospective_reused.jsonl"
FROZEN_FRESH384 = SOURCE_ROOT / "generated/prospective_fresh384.jsonl"
FROZEN_FRESH_EQUIV = SOURCE_ROOT / "generated/prospective_fresh_equiv.jsonl"
FROZEN_TELEMETRY = SOURCE_ROOT / "generated/telemetry.jsonl"
SOURCE_SHA = "245a452f80970485dd9d07dad560e35f04bbccc16f6147e36c98598b7426f106"
FRESH384_SHA = "74203f35df72c25d4b02f25605926fb34e36edc114cb0a4e2c2cd8f2771c84c6"
FRESH_EQUIV_SHA = "e1bc3b17c165b8aee210fc8267e4477c510abe467f47225c3c4f874f74005bab"
FROZEN_TELEMETRY_SHA = (
    "2885fc2cb685614ad084643c399776e707355daa2fdf17931a4f5856b8e19dc5"
)
EXPECTED_ROWS = 27858
NON_POLICY_SHA = "45e9b0a76879fd0e92f6536107073ecbe16a59832125e74fb6d12fb6230c527c"
BATCH_SHA = "a4e859f1340078e21c9ad2b4e0c04bac5c69c7463d8b8b859f83f79f915382fe"
TARGET_CHECKPOINT = Path(
    "/tmp/azlite_fresh_p1_parent_adapter/artifacts/step_0016/checkpoint.npz"
)
TARGET_CHECKPOINT_SHA = (
    "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34"
)
SUITE_SHA = "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04"
A16_SNAPSHOT_SHA = "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
INITIAL_OPTIMIZER_SHA = (
    "61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7"
)
SEED, GAMES, WORKERS, BASE = 45, 700, 24, 384
CONTEXTS = ("384:256", "1200:1200")
LANES = ("fresh384", "fixed768", "fixed1024", "fixed1280", "fixed1536", "fresh_equiv")
FIXED_BUDGETS = {
    "fixed768": 768,
    "fixed1024": 1024,
    "fixed1280": 1280,
    "fixed1536": 1536,
}

_EVALUATOR: CheckpointEvaluator | None = None


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def game_rng(game_index: int) -> random.Random:
    return random.Random(
        (SEED * 1_000_003)
        + game_index
        + (pr245.generation_worker_for(game_index) * 9_973)
    )


def inherited_mass(root: Node | None) -> int:
    return (
        0
        if root is None
        else int(sum(child.visit_count for child in root.children.values()))
    )


def search(
    game: KalahGame,
    row: dict[str, Any],
    rng: random.Random,
    *,
    root: Node | None,
    reuse: bool,
    simulations: int,
) -> tuple[list[float], Node, list[float]]:
    if _EVALUATOR is None:
        raise RuntimeError("worker evaluator is not initialized")
    options = row["teacher_search_profile"]["search_options"]
    noisy = bool(row["action_sampling_noise_enabled"])
    engine = PUCT(
        evaluator=_EVALUATOR,
        simulations=simulations,
        c_puct=float(row["teacher_search_profile"]["c_puct"]),
        rng=rng,
        root=root,
        fpu_mode=str(options["fpu_mode"]),
        reuse_subtree=reuse,
        normalize_values=bool(options["normalize_values"]),
        root_policy_mode=str(options["root_policy_mode"]),
        tactical_root_bias=float(options["tactical_root_bias"]),
        root_temperature=float(options.get("root_temperature", 0.0)),
    )
    visits, result_root = engine.run(
        game,
        dirichlet_alpha=float(row["dirichlet_alpha"]) if noisy else None,
        dirichlet_epsilon=float(row["target_dirichlet_epsilon"]) if noisy else 0.0,
    )
    prior = engine.root_summary()["root_prior_telemetry"]["after"]
    if prior is None:
        fail("missing root prior")
    return (
        build_policy_target(
            visits,
            legal_moves=row["legal_moves"],
            temperature=1.0 if int(row["move_index"]) < 10 else 0.1,
            mode=row["policy_target_mode"],
        ),
        result_root,
        [float(x) for x in prior],
    )


def init_worker(checkpoint: str) -> None:
    global _EVALUATOR
    _EVALUATOR = CheckpointEvaluator(Path(checkpoint), input_encoding="kalah_v3")


def reconstruct_game(
    item: tuple[int, list[dict[str, Any]]],
) -> tuple[int, list[dict[str, Any]]]:
    game_index, rows = item
    rows = sorted(rows, key=lambda row: int(row["move_index"]))
    if not rows or [int(row["move_index"]) for row in rows] != list(range(len(rows))):
        fail(f"non-contiguous plies game={game_index}")
    game, rng, reused_root = (
        KalahGame.from_state(standard_start_state()),
        game_rng(game_index),
        None,
    )
    records: list[dict[str, Any]] = []
    for ply, row in enumerate(rows):
        if encode_state(game.to_state(), input_encoding="kalah_v3") != row["state"]:
            fail(f"state mismatch game={game_index} ply={ply}")
        mass, before = inherited_mass(reused_root), rng.getstate()
        budgets = {
            "reused": BASE,
            "fresh384": BASE,
            **FIXED_BUDGETS,
            "fresh_equiv": BASE + mass,
        }
        runs = {}
        for lane, budget in budgets.items():
            clone = random.Random()
            clone.setstate(before)
            runs[lane] = (
                *search(
                    game,
                    row,
                    clone,
                    root=reused_root if lane == "reused" else None,
                    reuse=lane == "reused",
                    simulations=budget,
                ),
                clone,
            )
        priors = [run[2] for run in runs.values()]
        if not all(np.array_equal(priors[0], prior) for prior in priors[1:]):
            fail(f"root-noise mismatch game={game_index} ply={ply}")
        reused, warm_root, _prior, warm_rng = runs["reused"]
        if not np.array_equal(
            np.asarray(reused, dtype=np.float32),
            np.asarray(row["policy"], dtype=np.float32),
        ):
            fail(f"reused policy mismatch game={game_index} ply={ply}")
        if ply == 0 and (mass != 0 or runs["fresh384"][0] != runs["fresh_equiv"][0]):
            fail(f"ply-0 fresh geometry mismatch game={game_index}")
        move = sample_move(reused, row["legal_moves"], warm_rng)
        if not game.move(game.pit_index(move)):
            fail(f"illegal replay action game={game_index} ply={ply}")
        if (
            ply + 1 < len(rows)
            and encode_state(game.to_state(), input_encoding="kalah_v3")
            != rows[ply + 1]["state"]
        ):
            fail(f"sampled succession mismatch game={game_index} ply={ply}")
        records.append(
            {
                "game_index": game_index,
                "move_index": ply,
                "inherited_child_visit_mass": mass,
                "fresh_equiv_budget": BASE + mass,
                "policies": {lane: runs[lane][0] for lane in LANES + ("reused",)},
            }
        )
        rng, reused_root = warm_rng, warm_root.child_for_action(move)
    if (
        not game.over()
        or game.winner != rows[0]["winner"]
        or len(rows) != int(rows[0]["game_length"])
    ):
        fail(f"terminal mismatch game={game_index}")
    return game_index, records


def reconstruct(
    rows: list[dict[str, Any]], workers: int
) -> dict[tuple[int, int], dict[str, Any]]:
    games: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        games[int(row["game_index"])].append(row)
    if sorted(games) != list(range(GAMES)):
        fail("source is not the frozen 700-game replay")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(str(TARGET_CHECKPOINT),)
    ) as executor:
        completed = executor.map(reconstruct_game, sorted(games.items()), chunksize=1)
        records = [
            record for _index, game_records in completed for record in game_records
        ]
    return {(record["game_index"], record["move_index"]): record for record in records}


def non_policy_sha(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                {key: value for key, value in row.items() if key != "policy"}
                for row in rows
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def materialize_views(
    source: list[dict[str, Any]],
    reconstructed: dict[tuple[int, int], dict[str, Any]],
    workdir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Write source-ordered policy views and gate the two committed controls."""
    views = {
        lane: [
            {
                **copy.deepcopy(row),
                "policy": reconstructed[
                    (int(row["game_index"]), int(row["move_index"]))
                ]["policies"][lane],
            }
            for row in source
        ]
        for lane in LANES + ("reused",)
    }
    for lane, rows in views.items():
        write_jsonl(workdir / "derived" / f"{lane}.jsonl", rows)
    hashes = {
        lane: sha256_file(workdir / "derived" / f"{lane}.jsonl") for lane in views
    }
    if hashes["fresh384"] != FRESH384_SHA or hashes["fresh_equiv"] != FRESH_EQUIV_SHA:
        fail("reconstructed fresh control hash mismatch")
    return views, hashes


def load_derived_views(
    source: list[dict[str, Any]], workdir: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Load complete views without permitting a replay reconstruction."""
    views: dict[str, list[dict[str, Any]]] = {}
    for lane in LANES + ("reused",):
        path = workdir / "derived" / f"{lane}.jsonl"
        if not path.is_file():
            fail(f"resume missing derived view: {lane}")
        rows = read_jsonl(path)
        if len(rows) != len(source):
            fail(f"resume derived row count mismatch: {lane}")
        for index, (expected, actual) in enumerate(zip(source, rows, strict=True)):
            if "policy" not in actual:
                fail(f"resume derived policy missing: {lane} row={index}")
            if {key: value for key, value in actual.items() if key != "policy"} != {
                key: value for key, value in expected.items() if key != "policy"
            }:
                fail(
                    f"resume derived metadata or ordering mismatch: {lane} row={index}"
                )
        views[lane] = rows
    hashes = {
        lane: sha256_file(workdir / "derived" / f"{lane}.jsonl") for lane in views
    }
    if hashes["fresh384"] != FRESH384_SHA or hashes["fresh_equiv"] != FRESH_EQUIV_SHA:
        fail("resume fresh control hash mismatch")
    return views, hashes


def write_telemetry(
    path: Path,
    source: list[dict[str, Any]],
    records: dict[tuple[int, int], dict[str, Any]],
) -> None:
    write_jsonl(
        path,
        [records[(int(row["game_index"]), int(row["move_index"]))] for row in source],
    )


def load_telemetry(
    source: list[dict[str, Any]], views: dict[str, list[dict[str, Any]]], workdir: Path
) -> dict[tuple[int, int], dict[str, Any]]:
    """Load telemetry without replaying; fall back only to frozen PR #247 data."""
    path = workdir / "telemetry.jsonl"
    if not path.is_file():
        path = FROZEN_TELEMETRY
        if not path.is_file() or sha256_file(path) != FROZEN_TELEMETRY_SHA:
            fail("resume missing telemetry and frozen PR #247 telemetry is unavailable")
    records = read_jsonl(path)
    if len(records) != len(source):
        fail("resume telemetry row count mismatch")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for index, (source_row, record) in enumerate(zip(source, records, strict=True)):
        key = (int(source_row["game_index"]), int(source_row["move_index"]))
        if (
            int(record.get("game_index", -1)),
            int(record.get("move_index", -1)),
        ) != key:
            fail(f"resume telemetry ordering mismatch row={index}")
        mass = record.get("inherited_child_visit_mass")
        # The committed PR #247 telemetry predates this runner's concise field
        # name, so retain its explicit frozen schema during resume validation.
        fresh_equiv_budget = record.get(
            "fresh_equiv_budget", record.get("matched_fresh_equiv_simulations")
        )
        if not isinstance(mass, int) or mass < 0 or fresh_equiv_budget != BASE + mass:
            fail(f"resume telemetry budget mismatch row={index}")
        result[key] = {
            "game_index": key[0],
            "move_index": key[1],
            "inherited_child_visit_mass": mass,
            "fresh_equiv_budget": BASE + mass,
            "policies": {
                lane: views[lane][index]["policy"] for lane in LANES + ("reused",)
            },
        }
    if len(result) != len(source):
        fail("resume telemetry duplicate keys")
    return result


def target_geometry(
    rows: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
    targets: dict[str, list[list[float]]],
) -> dict[str, Any]:
    masses = np.asarray(
        [item["inherited_child_visit_mass"] for item in telemetry], dtype=float
    )
    edges = np.quantile(masses, [0.25, 0.5, 0.75])
    result: dict[str, Any] = {}
    means = []
    for fixed in FIXED_BUDGETS:
        comparisons, closer = {}, []
        for reference in ("fresh384", "fresh_equiv", "reused"):
            records = [
                isolation.policy_stats(a, b, row["legal_moves"])
                for row, a, b in zip(
                    rows, targets[fixed], targets[reference], strict=True
                )
            ]
            strata: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
            for record, mass in zip(records, masses, strict=True):
                strata[
                    f"inherited_mass_quartile:{np.searchsorted(edges, mass, side='right') + 1}"
                ].append(record)
            comparisons[f"{fixed}_vs_{reference}"] = {
                "overall": isolation.summarize(records),
                "by_inherited_mass_quartile": {
                    key: isolation.summarize(value)
                    for key, value in sorted(strata.items())
                },
            }
        for row, fixed_target, fresh, equiv in zip(
            rows,
            targets[fixed],
            targets["fresh384"],
            targets["fresh_equiv"],
            strict=True,
        ):
            closer.append(
                isolation.policy_stats(fixed_target, equiv, row["legal_moves"])["js"]
                < isolation.policy_stats(fixed_target, fresh, row["legal_moves"])["js"]
            )
        js = [
            isolation.policy_stats(target, equiv, row["legal_moves"])["js"]
            for row, target, equiv in zip(
                rows, targets[fixed], targets["fresh_equiv"], strict=True
            )
        ]
        means.append(float(np.mean(js)))
        result[fixed] = {
            "comparisons": comparisons,
            "mean_js_vs_fresh_equiv": means[-1],
            "fraction_closer_to_fresh_equiv_than_fresh384": float(np.mean(closer)),
        }
    result["monotonic_mean_js_vs_fresh_equiv_decreasing"] = all(
        right <= left for left, right in zip(means, means[1:], strict=True)
    )
    return result


def convergence_curves(
    rows: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
    targets: dict[str, list[list[float]]],
) -> dict[str, Any]:
    masses = np.asarray(
        [item["inherited_child_visit_mass"] for item in telemetry], dtype=float
    )
    edges = np.quantile(masses, [0.25, 0.5, 0.75])
    result: dict[str, Any] = {}
    for lane in FIXED_BUDGETS:
        per_state, groups = [], defaultdict(list)
        for row, mass, target, equiv in zip(
            rows, masses, targets[lane], targets["fresh_equiv"], strict=True
        ):
            js = isolation.policy_stats(target, equiv, row["legal_moves"])["js"]
            per_state.append(
                {
                    "game_index": int(row["game_index"]),
                    "move_index": int(row["move_index"]),
                    "js_vs_fresh_equiv": js,
                }
            )
            groups[
                f"inherited_mass_quartile:{np.searchsorted(edges, mass, side='right') + 1}"
            ].append(js)
            groups[f"phase:{isolation.phase(row)}"].append(js)
            groups[f"legal_move_count:{len(row['legal_moves'])}"].append(js)
            groups[
                f"noise_enabled:{bool(row['action_sampling_noise_enabled'])}"
            ].append(js)
        result[lane] = {
            "per_state_js_vs_fresh_equiv": per_state,
            "mean_js_by_stratum": {
                key: float(np.mean(values)) for key, values in sorted(groups.items())
            },
        }
    return result


def cost_accounting(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    equiv = np.asarray([item["fresh_equiv_budget"] for item in telemetry], dtype=float)

    def distribution(values: np.ndarray) -> dict[str, float]:
        return {
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        }

    expected = {"fresh384": BASE, **FIXED_BUDGETS}
    return {
        "expected_target_total_and_multiplier": {
            lane: {
                "target_budget": budget,
                "total_budget": BASE + budget,
                "target_multiplier_vs_fresh384": budget / BASE,
                "total_multiplier_vs_fresh384": (BASE + budget) / (2 * BASE),
            }
            for lane, budget in expected.items()
        },
        "fresh_equiv_empirical": {
            "target_budget": distribution(equiv),
            "total_budget": distribution(BASE + equiv),
            "target_multiplier_vs_fresh384": distribution(equiv / BASE),
            "total_multiplier_vs_fresh384": distribution((BASE + equiv) / (2 * BASE)),
        },
    }


def load_cached_lane(
    workdir: Path, lane: str
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]] | None:
    checkpoints = {
        step: workdir / "train" / lane / f"step_{step:04d}.pt"
        for step in replay_train.STEPS
    }
    exported = workdir / "train" / lane / "step_0016" / "checkpoint.npz"
    if not all(path.is_file() for path in (*checkpoints.values(), exported)):
        return None
    snapshots, optimizers = {}, {}
    for step, path in checkpoints.items():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or set(saved) != {"model", "optimizer"}:
            fail(f"resume invalid training checkpoint: {lane} step={step}")
        if not isinstance(saved["model"], dict) or not isinstance(
            saved["optimizer"], dict
        ):
            fail(f"resume invalid training checkpoint state: {lane} step={step}")
        snapshots[step], optimizers[step] = saved["model"], saved["optimizer"]
    return snapshots, optimizers


def train_lanes(
    rows: dict[str, list[dict[str, Any]]],
    model: dict[str, torch.Tensor],
    optimizer: dict[str, Any],
    parent: dict[str, torch.Tensor],
    workdir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    initial, result, artifacts = replay_train.optimizer_state_sha256(optimizer), {}, {}
    targets = {lane: [row["policy"] for row in view] for lane, view in rows.items()}
    trained: set[str] = set()
    for lane in LANES:
        cached = load_cached_lane(workdir, lane)
        if cached is None:
            snapshots, optimizers, invocation = train_mechanism.run_lane(
                rows[lane], model, optimizer, parent
            )
            trained.add(lane)
            for step, state in snapshots.items():
                output = workdir / "train" / lane / f"step_{step:04d}"
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": state, "optimizer": optimizers[step]},
                    output.with_suffix(".pt"),
                )
                if step == 16:
                    artifacts[lane] = export(state, output, f"pr247_{lane}")
        else:
            snapshots, optimizers = cached
            invocation = {"before": initial, "after": initial}
            artifact = workdir / "train" / lane / "step_0016" / "artifact"
            artifacts[lane] = (
                artifact
                if artifact.is_dir()
                else export(snapshots[16], artifact.parent, f"pr247_{lane}")
            )
        result[lane] = {
            "optimizer_invocation": invocation,
            "metrics": replay_train.metrics(
                rows[lane], snapshots, parent, model, copy.deepcopy(optimizer)
            ),
            "checkpoints": {},
            "cross_target_ce": {},
        }
        for step, state in snapshots.items():
            result[lane]["checkpoints"][str(step)] = train_mechanism.checkpoint_record(
                state, optimizers[step]
            )
            result[lane]["cross_target_ce"][str(step)] = isolation.all_target_metrics(
                rows[lane], state, targets
            )
    repeated = (
        train_mechanism.repeated_lane_check(
            "fixed1024", rows["fixed1024"], model, optimizer, parent
        )
        if "fixed1024" in trained
        else True
    )
    if not repeated or replay_train.optimizer_state_sha256(optimizer) != initial:
        fail("fixed1024 identity or immutable optimizer invariant")
    result["optimizer_invariants"] = {
        "initial_sha256": initial,
        "pristine_unchanged": replay_train.optimizer_state_sha256(optimizer) == initial,
        "repeated_fixed1024": repeated,
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
        for left in (*FIXED_BUDGETS, "fresh_equiv")
        for right in ("fresh384", "fresh_equiv")
        if left != right
    }
    return result


def selection_pass_table(
    training: dict[str, Any], arena: dict[str, Any]
) -> dict[str, Any]:
    contrasts = arena["paired_contrasts"]
    result = {}
    for lane in FIXED_BUDGETS:
        versus_fresh = contrasts[f"{lane}_minus_fresh384"]["1200:1200"]
        versus_equiv = contrasts[f"{lane}_minus_fresh_equiv"]["1200:1200"]
        fit = training[lane]["metrics"]["16"]["fit_fraction"] >= 0.25
        lower_positive = versus_fresh["opening_bootstrap_ci"]["lower_95"] > 0.0
        close = (
            abs(
                arena[lane]["1200:1200"]["effect"]
                - arena["fresh_equiv"]["1200:1200"]["effect"]
            )
            <= 0.02
        )
        includes_zero = (
            versus_equiv["opening_bootstrap_ci"]["lower_95"]
            <= 0.0
            <= versus_equiv["opening_bootstrap_ci"]["upper_95"]
        )
        result[lane] = {
            "fit_fraction_at_least_025": fit,
            "paired_vs_fresh384_lower_95_positive": lower_positive,
            "absolute_effect_distance_to_fresh_equiv_at_most_002": close,
            "paired_vs_fresh_equiv_ci_includes_zero": includes_zero,
            "passes": fit and lower_positive and close and includes_zero,
        }
    return result


def compute_efficiency_curve(arena: dict[str, Any]) -> dict[str, Any]:
    fresh = arena["fresh384"]["1200:1200"]["effect"]
    result, previous = {}, fresh
    for lane, budget in FIXED_BUDGETS.items():
        effect = arena[lane]["1200:1200"]["effect"]
        result[lane] = {
            "target_budget": budget,
            "effect": effect,
            "gain_vs_fresh384": effect - fresh,
            "marginal_gain_per_plus_256": (effect - previous) / ((budget - BASE) / 256),
        }
        previous = effect
    return result


def classify(
    invariants: bool,
    geometry: dict[str, Any],
    training: dict[str, Any],
    arena: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not invariants:
        return "invariant_failure", {}, {}
    passes, curve = (
        selection_pass_table(training, arena),
        compute_efficiency_curve(arena),
    )
    selected = [lane for lane in FIXED_BUDGETS if passes[lane]["passes"]]
    if selected:
        first = selected[0]
        if first == "fixed768":
            return "fixed_768_sufficient", passes, curve
        if first == "fixed1024":
            return "fixed_1024_sufficient", passes, curve
        high = ["fixed1280", "fixed1536"]
        if all(passes[lane]["passes"] for lane in high):
            return "fixed_target_budget_retains_gain", passes, curve
        return "only_high_fixed_budget_matches", passes, curve
    fresh_equiv_reproduces = (
        arena["paired_contrasts"]["fresh_equiv_minus_fresh384"]["1200:1200"][
            "opening_bootstrap_ci"
        ]["lower_95"]
        > 0.0
    )
    if fresh_equiv_reproduces:
        return "adaptive_budget_materially_better", passes, curve
    if geometry["monotonic_mean_js_vs_fresh_equiv_decreasing"]:
        return "fixed_compute_effect_not_monotonic", passes, curve
    return "fixed_compute_effect_not_monotonic", passes, curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr247_fixed_target_budget")
    )
    parser.add_argument("--replay", type=Path, default=SOURCE)
    parser.add_argument("--fresh384", type=Path, default=FROZEN_FRESH384)
    parser.add_argument("--fresh-equiv", type=Path, default=FROZEN_FRESH_EQUIV)
    parser.add_argument(
        "--canonical-suite",
        type=Path,
        default=Path("/tmp/azlite_opening_suite/medium_eval.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument(
        "--resume-derived",
        action="store_true",
        help="Reuse validated derived views and persisted telemetry without replaying games.",
    )
    args = parser.parse_args()
    if (
        args.workers != WORKERS
        or sha256_file(args.replay) != SOURCE_SHA
        or sha256_file(args.fresh384) != FRESH384_SHA
        or sha256_file(args.fresh_equiv) != FRESH_EQUIV_SHA
        or sha256_file(args.canonical_suite) != SUITE_SHA
        or sha256_file(TARGET_CHECKPOINT) != TARGET_CHECKPOINT_SHA
    ):
        fail(
            "frozen source, fresh replay, suite, evaluator, or worker invariant mismatch"
        )
    source = read_jsonl(args.replay)
    if args.resume_derived:
        complete_views, replay_hashes = load_derived_views(source, args.workdir)
        reconstructed = load_telemetry(source, complete_views, args.workdir)
    else:
        reconstructed = reconstruct(source, args.workers)
        complete_views, replay_hashes = materialize_views(
            source, reconstructed, args.workdir
        )
        write_telemetry(args.workdir / "telemetry.jsonl", source, reconstructed)
    blocked, groups = replay_train.exclusion_hashes(args.canonical_suite)
    eligible, exclusions = replay_train.filter_rows(source, blocked, groups)
    if len(eligible) != EXPECTED_ROWS:
        fail("eligible row count mismatch")
    eligible_keys = {
        (int(row["game_index"]), int(row["move_index"])) for row in eligible
    }
    views = {
        lane: [
            row
            for row in complete_views[lane]
            if (int(row["game_index"]), int(row["move_index"])) in eligible_keys
        ]
        for lane in LANES
    }
    invariants = isolation.assert_views(eligible, views)
    non_policy_hashes = {lane: non_policy_sha(rows) for lane, rows in views.items()}
    if (
        invariants["batch_plan_sha256"] != BATCH_SHA
        or len(eligible) != EXPECTED_ROWS
        or set(non_policy_hashes.values()) != {NON_POLICY_SHA}
    ):
        fail("row metadata or batch-plan hash mismatch")
    invariants["non_policy_state_outcome_sha256"] = non_policy_hashes
    telemetry = [
        reconstructed[(int(row["game_index"]), int(row["move_index"]))]
        for row in eligible
    ]
    targets = {
        lane: [record["policies"][lane] for record in telemetry]
        for lane in LANES + ("reused",)
    }
    geometry = target_geometry(eligible, telemetry, targets)
    convergence = convergence_curves(eligible, telemetry, targets)
    snapshot = torch.load(
        replay_train.A16_SNAPSHOT, map_location="cpu", weights_only=False
    )
    model, optimizer = replay_train.immutable_initial_state(snapshot)
    if (
        sha256_file(replay_train.A16_SNAPSHOT) != A16_SNAPSHOT_SHA
        or replay_train.optimizer_state_sha256(optimizer) != INITIAL_OPTIMIZER_SHA
    ):
        fail("A16 snapshot or initial optimizer hash mismatch")
    p1 = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1, replay_train.P1_CHECKPOINT)
    parent = {
        name: value.detach().cpu().clone() for name, value in p1.state_dict().items()
    }
    training, artifacts = train_lanes(views, model, optimizer, parent, args.workdir)
    result: dict[str, Any] = {
        "schema": "pr247_fixed_target_budget_v1",
        "classification": "inconclusive",
        "frozen_artifacts": {
            "prospective_reused_sha256": SOURCE_SHA,
            "fresh384_sha256": FRESH384_SHA,
            "fresh_equiv_sha256": FRESH_EQUIV_SHA,
            "a16_snapshot_sha256": A16_SNAPSHOT_SHA,
            "initial_optimizer_sha256": INITIAL_OPTIMIZER_SHA,
        },
        "derived_replay_sha256": replay_hashes,
        "exclusions": exclusions,
        "invariants": invariants,
        "cost_accounting": cost_accounting(telemetry),
        "target_geometry": geometry,
        "convergence_curves": convergence,
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
        classification, passes, curve = classify(
            True, geometry, training, result["canonical_arena"]
        )
        result["classification"] = classification
        result["selection_pass_table"] = passes
        result["compute_efficiency_curve"] = curve
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
