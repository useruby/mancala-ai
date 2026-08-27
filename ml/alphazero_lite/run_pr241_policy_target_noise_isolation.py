#!/usr/bin/env python3
# ruff: noqa: E402
"""Isolate root Dirichlet noise in PR #241's fixed ordinary replay targets.

This intentionally never calls self-play.  It reconstructs independent target
searches for the already frozen eligible rows, changing only ``policy``.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import hashlib
import json
import multiprocessing
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

from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.arena import canonical_game_state_hash
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as shadow_replay
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_onpolicy_shadow_replay import (
    A16_SNAPSHOT,
    P1_CHECKPOINT,
    batches,
    exclusion_hashes,
    filter_rows,
    metrics,
    policy,
    train_lane,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    export,
    new_model,
)
from ml.alphazero_lite.run_fresh_p1_adapter_matched_q_feedback import (
    decode_kalah_v3_base_state,
)
from ml.alphazero_lite.self_play import CheckpointEvaluator, PUCT, build_policy_target
from ml.alphazero_lite.train import load_checkpoint_into_model

SOURCE_SHA = "6671e248af4a4c82e1155c798cb7490cd66cd80dc10b203c97d89dced94527f2"
SUITE_SHA = "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04"
TARGET_NAMESPACE = "pr241_policy_target_noise_isolation_v1"
CONTEXTS = ("384:256", "1200:1200")
A16_TARGET_CHECKPOINT = Path(
    "/tmp/azlite_fresh_p1_parent_adapter/artifacts/step_0016/checkpoint.npz"
)
A16_TARGET_CHECKPOINT_SHA = (
    "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34"
)

# Linux workers are forked after the immutable rows are loaded, so they share
# the large replay copy-on-write rather than serializing 27k full row objects.
_TARGET_ROWS: list[dict[str, Any]] = []
_TARGET_EVALUATOR: CheckpointEvaluator | None = None


def canonical_state_hash(row: dict[str, Any]) -> str:
    state = decode_kalah_v3_base_state(list(row["state"]))
    return canonical_game_state_hash(KalahGame.from_state(state))


def target_seed(row: dict[str, Any]) -> int:
    """Seed contract: namespace, game index, move index, canonical state hash only."""
    identity = [
        TARGET_NAMESPACE,
        int(row["game_index"]),
        int(row["move_index"]),
        canonical_state_hash(row),
    ]
    return int(
        hashlib.sha256(
            json.dumps(identity, separators=(",", ":")).encode("ascii")
        ).hexdigest()[:16],
        16,
    ) % (2**31)


def phase(row: dict[str, Any]) -> str:
    state = decode_kalah_v3_base_state(list(row["state"]))
    stones = sum(state["player_pits"]) + sum(state["opponent_pits"])
    return "opening" if stones > 24 else "midgame" if stones > 12 else "late"


def search_target(
    evaluator: CheckpointEvaluator,
    row: dict[str, Any],
    *,
    noisy: bool,
    simulations: int = 384,
) -> list[float]:
    game = KalahGame.from_state(decode_kalah_v3_base_state(list(row["state"])))
    enabled = bool(row["action_sampling_noise_enabled"])
    search = PUCT(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=1.25,
        rng=random.Random(target_seed(row)),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
    )
    visits, _ = search.run(
        game,
        dirichlet_alpha=0.3 if noisy and enabled else None,
        dirichlet_epsilon=0.3 if noisy and enabled else 0.0,
    )
    return build_policy_target(
        visits,
        legal_moves=[int(move) for move in row["legal_moves"]],
        # PR #241 inherited the fresh-P1 generation defaults: 1.0 through ply 9,
        # then 0.1. This affects only visit-to-policy conversion, not gameplay.
        temperature=1.0 if int(row["move_index"]) < 10 else 0.1,
        mode=str(row["policy_target_mode"]),
    )


def initialize_target_worker(checkpoint: str) -> None:
    global _TARGET_EVALUATOR
    _TARGET_EVALUATOR = CheckpointEvaluator(Path(checkpoint), input_encoding="kalah_v3")


def reconstruct_target_pair(index: int) -> tuple[list[float], list[float]]:
    if _TARGET_EVALUATOR is None:
        raise RuntimeError("target worker evaluator was not initialized")
    row = _TARGET_ROWS[index]
    return (
        search_target(_TARGET_EVALUATOR, row, noisy=True),
        search_target(_TARGET_EVALUATOR, row, noisy=False),
    )


def reconstruct_clean1200(index: int) -> tuple[int, list[float]]:
    if _TARGET_EVALUATOR is None:
        raise RuntimeError("target worker evaluator was not initialized")
    return index, search_target(
        _TARGET_EVALUATOR, _TARGET_ROWS[index], noisy=False, simulations=1200
    )


def parallel_targets(
    rows: list[dict[str, Any]], checkpoint: Path, workers: int
) -> tuple[list[list[float]], list[list[float]], dict[int, list[float]]]:
    global _TARGET_ROWS
    _TARGET_ROWS = rows
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=initialize_target_worker,
        initargs=(str(checkpoint),),
    ) as executor:
        pairs = list(
            executor.map(reconstruct_target_pair, range(len(rows)), chunksize=16)
        )
        probe_indexes = [
            index
            for _state_hash, index in sorted(
                {
                    canonical_state_hash(row): index for index, row in enumerate(rows)
                }.items()
            )[:256]
        ]
        clean = dict(executor.map(reconstruct_clean1200, probe_indexes, chunksize=4))
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs], clean


def policy_stats(
    left: list[float], right: list[float], legal: list[int]
) -> dict[str, float | bool]:
    a, b = np.asarray(left, dtype=float)[legal], np.asarray(right, dtype=float)[legal]
    midpoint = (a + b) / 2
    a_positive, b_positive = a > 0, b > 0
    return {
        "l1": float(np.abs(a - b).sum()),
        "js": float(
            (
                np.sum(a[a_positive] * np.log(a[a_positive] / midpoint[a_positive]))
                + np.sum(b[b_positive] * np.log(b[b_positive] / midpoint[b_positive]))
            )
            / 2
        ),
        "top1_disagreement": int(legal[np.argmax(a)] != legal[np.argmax(b)]),
        "entropy_difference": float(
            -np.sum(b[b_positive] * np.log(b[b_positive]))
            + np.sum(a[a_positive] * np.log(a[a_positive]))
        ),
    }


def summarize(records: list[dict[str, float | bool]]) -> dict[str, float]:
    if not records:
        return {}
    return {
        "mean_legal_policy_l1": float(np.mean([r["l1"] for r in records])),
        "p50_legal_policy_l1": float(np.percentile([r["l1"] for r in records], 50)),
        "p90_legal_policy_l1": float(np.percentile([r["l1"] for r in records], 90)),
        "p99_legal_policy_l1": float(np.percentile([r["l1"] for r in records], 99)),
        "mean_js": float(np.mean([r["js"] for r in records])),
        "top1_disagreement": float(np.mean([r["top1_disagreement"] for r in records])),
        "target_entropy_difference": float(
            np.mean([r["entropy_difference"] for r in records])
        ),
    }


def target_diagnostics(
    rows: list[dict[str, Any]], targets: dict[str, list[list[float]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left, right in (
        ("original_noisy", "fresh_noisy"),
        ("fresh_noisy", "fresh_denoised"),
        ("original_noisy", "fresh_denoised"),
    ):
        records = [
            policy_stats(a, b, row["legal_moves"])
            for row, a, b in zip(rows, targets[left], targets[right], strict=True)
        ]
        groups: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
        for row, record in zip(rows, records, strict=True):
            groups[f"root_noise:{bool(row['action_sampling_noise_enabled'])}"].append(
                record
            )
            groups[f"phase:{phase(row)}"].append(record)
            groups[f"legal_move_count:{len(row['legal_moves'])}"].append(record)
        result[f"{left}_vs_{right}"] = {
            "overall": summarize(records),
            "by_stratum": {
                key: summarize(value) for key, value in sorted(groups.items())
            },
        }
    return result


def assert_views(
    source: list[dict[str, Any]], views: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    source_plan = [item.tolist() for item in batches(source)]
    result = {
        "eligible_rows": len(source),
        "batch_plan_sha256": digest(source_plan),
        "views": {},
    }
    for name, rows in views.items():
        if len(rows) != len(source):
            raise RuntimeError("row count mismatch")
        for original, derived in zip(source, rows, strict=True):
            if any(
                original[key] != derived[key] for key in original if key != "policy"
            ):
                raise RuntimeError(f"non-policy label changed in {name}")
            legal = derived["legal_moves"]
            target = np.asarray(derived["policy"], dtype=float)
            if (
                not np.isclose(target.sum(), 1.0)
                or np.any(target < 0)
                or any(target[i] != 0 for i in range(6) if i not in legal)
            ):
                raise RuntimeError(f"illegal policy target in {name}")
        result["views"][name] = {
            "policy_sha256": digest([row["policy"] for row in rows]),
            "batch_plan_sha256": digest([item.tolist() for item in batches(rows)]),
        }
        if result["views"][name]["batch_plan_sha256"] != result["batch_plan_sha256"]:
            raise RuntimeError("batch plan mismatch")
    return result


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def all_target_metrics(
    rows: list[dict[str, Any]],
    state: dict[str, torch.Tensor],
    targets: dict[str, list[list[float]]],
) -> dict[str, float]:
    predicted = policy(state, rows)
    return {
        name: float(
            np.mean(
                -np.sum(
                    np.asarray(values) * np.log(np.maximum(predicted, 1e-12)), axis=1
                )
            )
        )
        for name, values in targets.items()
    }


def arena_records(
    workdir: Path,
    challenger: Path,
    p1: Path,
    context: str,
    role: str,
    workers: int,
    suite: Path,
) -> list[dict[str, Any]]:
    # This is the PR #241 ordinary arena helper; shadow is explicitly absent.
    from ml.alphazero_lite import run_fresh_p1_adapter_lagged_parent_shadow_q as arena

    arena.ARENA_SUITE = suite
    return arena.arena_records(workdir, challenger, p1, None, context, role, workers)


def arena_summary(
    workdir: Path,
    artifacts: dict[str, dict[int, Path]],
    p1: Path,
    workers: int,
    suite: Path,
) -> dict[str, Any]:
    controls = {
        context: arena_records(
            workdir / "arena", p1, p1, context, "p1_control", workers, suite
        )
        for context in CONTEXTS
    }
    result: dict[str, Any] = {}
    for lane, checkpoints in artifacts.items():
        result[lane] = {}
        for step, artifact in checkpoints.items():
            result[lane][str(step)] = {}
            for context, control in controls.items():
                effect = paired_opening_candidate_effect(
                    arena_records(
                        workdir / "arena",
                        artifact,
                        p1,
                        context,
                        f"{lane}_{step}",
                        workers,
                        suite,
                    ),
                    control,
                )
                ci = effect["opening_bootstrap_ci"]
                result[lane][str(step)][context] = {
                    "effect": effect["paired_candidate_effect"],
                    "ci": ci,
                    "safe": ci["upper_95"] >= 0 or ci["lower_95"] >= -0.03,
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr241_policy_target_noise")
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
        help="Development-only: omit the required ordinary-PUCT arena and P0 gate.",
    )
    args = parser.parse_args()
    if (
        sha256_file(args.replay) != SOURCE_SHA
        or sha256_file(args.canonical_suite) != SUITE_SHA
    ):
        raise RuntimeError("frozen replay or canonical suite hash mismatch")
    source, exclusion = filter_rows(
        read_jsonl(args.replay), *exclusion_hashes(args.canonical_suite)
    )
    initial = torch.load(A16_SNAPSHOT, map_location="cpu", weights_only=False)
    shadow_replay.INITIAL_OPTIMIZER = initial["optimizer"]
    p1_model = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1_model, P1_CHECKPOINT)
    parent = {
        key: value.detach().cpu().clone()
        for key, value in p1_model.state_dict().items()
    }
    # The original lane must be reproduced before any reconstructed target is trusted.
    reproduced, _ = train_lane(source, initial, parent, torch.device("cpu"))
    historical = Path("/tmp/azlite_onpolicy_shadow_replay/ordinary_onpolicy_train")
    reproduction = {
        str(step): all(
            torch.equal(
                value,
                torch.load(
                    historical / f"step_{step:04d}.pt",
                    map_location="cpu",
                    weights_only=False,
                )["model"][key],
            )
            for key, value in state.items()
        )
        for step, state in reproduced.items()
    }
    if not all(reproduction.values()):
        raise RuntimeError("invariant_failure: PR #241 original lane did not reproduce")
    if sha256_file(A16_TARGET_CHECKPOINT) != A16_TARGET_CHECKPOINT_SHA:
        raise RuntimeError("A16 target evaluator hash mismatch")
    fresh_noisy, fresh_denoised, clean_targets = parallel_targets(
        source, A16_TARGET_CHECKPOINT, args.workers
    )
    targets = {
        "original_noisy": [row["policy"] for row in source],
        "fresh_noisy": fresh_noisy,
        "fresh_denoised": fresh_denoised,
    }
    views = {
        name: [
            {**copy.deepcopy(row), "policy": target}
            for row, target in zip(source, values, strict=True)
        ]
        for name, values in targets.items()
    }
    invariants = assert_views(source, views)
    for name, rows in views.items():
        write_jsonl(args.workdir / f"{name}.jsonl", rows)
    # Freeze the probe by canonical state hash before calculating its clean targets.
    probe_indexes = sorted(
        {canonical_state_hash(row): index for index, row in enumerate(source)}.items()
    )[:256]
    probe = []
    for _state_hash, index in probe_indexes:
        row = source[index]
        clean = clean_targets[index]
        probe.append(
            {
                "state_hash": canonical_state_hash(row),
                "row_index": index,
                "clean1200": clean,
                "comparisons": {
                    name: policy_stats(values[index], clean, row["legal_moves"])
                    for name, values in targets.items()
                },
            }
        )
    clean_probe = {
        "state_hashes_sha256": digest([row["state_hash"] for row in probe]),
        "count": len(probe),
        "against_clean1200": {
            name: summarize([row["comparisons"][name] for row in probe])
            for name in targets
        },
        "records": probe,
    }
    lanes, artifacts, training = {}, {}, {}
    for name, rows in views.items():
        snapshots, optimizers = train_lane(rows, initial, parent, torch.device("cpu"))
        lanes[name] = snapshots
        artifacts[name] = {}
        training[name] = {
            "lane_metrics": metrics(rows, snapshots, parent, initial["model"]),
            "cross_target_ce": {},
        }
        for step, state in snapshots.items():
            out = args.workdir / f"{name}_train" / f"step_{step:04d}"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": state, "optimizer": optimizers[step]}, out.with_suffix(".pt")
            )
            artifacts[name][step] = export(state, out, f"pr241_{name}_{step}")
            training[name]["cross_target_ce"][str(step)] = all_target_metrics(
                source, state, targets
            )
    result: dict[str, Any] = {
        "schema": TARGET_NAMESPACE,
        "classification": "inconclusive",
        "source": {
            "sha256": SOURCE_SHA,
            "row_count": 27642,
            "eligible_row_count": len(source),
            "exclusions": exclusion,
        },
        "seed_contract": "sha256([namespace, game_index, move_index, canonical_state_hash])[:16] mod 2**31; shared by fresh_noisy and fresh_denoised",
        "reproduction": reproduction,
        "invariants": invariants,
        "target_differences": target_diagnostics(source, targets),
        "clean1200_probe": clean_probe,
        "training": training,
    }
    if not args.skip_arena:
        p1_artifact = P1_CHECKPOINT.parent / "artifact"
        result["ordinary_puct_arena"] = arena_summary(
            args.workdir, artifacts, p1_artifact, args.workers, args.canonical_suite
        )
        d = result["ordinary_puct_arena"]["fresh_denoised"]["16"]
        n = result["ordinary_puct_arena"]["fresh_noisy"]["16"]
        d_fit = training["fresh_denoised"]["lane_metrics"]["16"]["fit_fraction"] >= 0.25
        d_safe = all(d[context]["safe"] for context in CONTEXTS)
        if d_fit and d_safe:
            result["p0_gate_eligible"] = True
            # Direct P0 gate follows the same ordinary arena contract and is not promotion.
            p0 = REPO_ROOT / "model-artifact/current"
            result["p0_gate"] = {
                context: paired_opening_candidate_effect(
                    arena_records(
                        args.workdir / "p0_gate",
                        artifacts["fresh_denoised"][16],
                        p0,
                        context,
                        "fresh_denoised_16_vs_p0",
                        args.workers,
                        args.canonical_suite,
                    ),
                    arena_records(
                        args.workdir / "p0_gate",
                        p0,
                        p0,
                        context,
                        "p0_control",
                        args.workers,
                        args.canonical_suite,
                    ),
                )
                for context in CONTEXTS
            }
        else:
            result["p0_gate_eligible"] = False
        low_improved = d["384:256"]["effect"] > n["384:256"]["effect"]
        high_preserved = d["1200:1200"]["effect"] >= n["1200:1200"]["effect"]
        if d_fit and d_safe and low_improved and high_preserved:
            result["classification"] = "denoised_targets_resolve_budget_split"
        elif low_improved and not (d_safe and high_preserved):
            result["classification"] = "target_noise_drives_low_budget_regression"
        elif low_improved and not high_preserved:
            result["classification"] = "denoising_sacrifices_high_budget_gain"
        elif not low_improved and high_preserved:
            result["classification"] = "policy_target_noise_not_causal"
        else:
            result["classification"] = "inconclusive"
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
