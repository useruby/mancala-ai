#!/usr/bin/env python3
# ruff: noqa: E402
"""Repeat PR #258's frozen 8,192-example plans once, without resetting Adam."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import consumed_suite_registry as registry_module
from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as replay
from ml.alphazero_lite import run_pr241_optimizer_isolation_reproduction as contract
from ml.alphazero_lite import run_pr258_two_replay_aggregation as pr258
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import sha256_file
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    ADAPTER_KEYS,
    export,
    new_model,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (
    incumbent_policy_batch,
    mixed_policy_target,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch, _losses

SOURCE = Path("/tmp/azlite_pr258_two_replay_aggregation")
SUITE_SEEDS = {"Y": 25042, "Z": 26042, "AA": 27042}
CHECKPOINTS = (16, 32)
TELEMETRY_STEPS = (1, 4, 16, 32)


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


def adapter_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return all(torch.equal(left[key], right[key]) for key in ADAPTER_KEYS)


def verify_source(
    registry: dict[str, registry_module.ConsumedSuite],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    """Load the committed replay rows and plans; never derive a replacement plan."""
    frozen = load_json(SOURCE / "frozen_replays_and_plans.json")
    candidates = load_json(SOURCE / "frozen_candidates.json")
    registry_module.validate(registry)
    if list(registry) != ["canonical", *"ABCDEFGHIJKLMNOPQRSTUVWX"]:
        fail("registry is not canonical through X")
    source_registry = load_json(SOURCE / "suite_registry.json")
    source_suites = {**source_registry["consumed"], **source_registry["newly_consumed"]}
    for label in ("V", "W", "X"):
        source_suites[label].pop("openings")
    if registry_module.manifest(registry) != source_suites:
        fail("PR #258 consumed-suite manifest mismatch")
    blocked, groups = replay.exclusion_hashes(pr258.pr249.CANONICAL_SUITE)
    consumed = set().union(
        *(replay.canonical_arena_hashes(spec.path) for spec in registry.values())
    )
    groups = {**groups, "consumed_evaluation_openings": consumed}
    blocked |= consumed
    replays: dict[int, list[dict[str, Any]]] = {}
    for seed in pr258.SEEDS:
        path = SOURCE / f"seed{seed}/generated/ordinary_reused.jsonl"
        raw = load_rows(path)
        eligible, exclusions = replay.filter_rows(raw, blocked, groups)
        expected = frozen["replays"][str(seed)]
        actual = {
            "raw_rows": len(raw),
            "eligible_rows": len(eligible),
            "replay_sha256": sha256_file(path),
            "state_outcome_sha256": pr258.canonical_sha(
                [
                    {key: value for key, value in row.items() if key != "policy"}
                    for row in eligible
                ]
            ),
            "exclusions": exclusions,
        }
        if actual != expected:
            fail(f"PR #258 replay mismatch for seed {seed}")
        replays[seed] = eligible
    return replays, frozen, candidates


def lanes_from_manifest(
    replays: dict[int, list[dict[str, Any]]], frozen: dict[str, Any]
) -> dict[
    str, dict[str, tuple[list[dict[str, Any]], list[list[int]], list[dict[str, Any]]]]
]:
    result = {}
    for pair_index, (seed_a, seed_b) in enumerate(pr258.PAIRS, 1):
        pair = f"pair{pair_index}"
        manifest = frozen["sample_plans"][pair]
        plans = {name: value["batches"] for name, value in manifest["plans"].items()}
        for name, batches in plans.items():
            if (
                pr258.canonical_sha(batches) != manifest["plans"][name]["sha256"]
                or len(batches) != 16
                or any(len(batch) != 512 for batch in batches)
                or len({index for batch in batches for index in batch}) != 8192
            ):
                fail(f"PR #258 sample plan mismatch for {pair} {name}")
        a_order = [index for batch in plans["single_a"] for index in batch]
        b_order = [index for batch in plans["single_b"] for index in batch]
        pair_rows = [replays[seed_a][index] for index in a_order[:4096]] + [
            replays[seed_b][index] for index in b_order[:4096]
        ]
        validation = manifest["validation_games"]
        result[pair] = {
            "single_a": (
                replays[seed_a],
                plans["single_a"],
                [
                    row
                    for row in replays[seed_a]
                    if row["game_index"] in validation["a"]
                ],
            ),
            "single_b": (
                replays[seed_b],
                plans["single_b"],
                [
                    row
                    for row in replays[seed_b]
                    if row["game_index"] in validation["b"]
                ],
            ),
            "pair_mix": (
                pair_rows,
                plans["pair_mix"],
                [row for row in replays[seed_a] if row["game_index"] in validation["a"]]
                + [
                    row
                    for row in replays[seed_b]
                    if row["game_index"] in validation["b"]
                ],
            ),
        }
    return result


def train_twice(
    rows: list[dict[str, Any]],
    batches: list[list[int]],
    a16: dict[str, torch.Tensor],
    adam: dict[str, Any],
    parent: dict[str, torch.Tensor],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]]:
    """Execute epoch two as the exact, contiguous repetition of epoch one."""
    model = new_model(torch.device("cpu"))
    model.load_state_dict(copy.deepcopy(a16))
    pr258.apply_trainable_scope(model, "policy_adapter_only")
    optimizer = replay.load_isolated_optimizer(model, adam)
    parent_model = new_model(torch.device("cpu"))
    parent_model.load_state_dict(parent)
    for parameter in parent_model.parameters():
        parameter.requires_grad_(False)
    snapshots, optimizers = {}, {}
    model.train()
    for step, indexes in enumerate(batches + batches, 1):
        batch = _batch(rows, np.asarray(indexes, dtype=np.int64), torch.device("cpu"))
        target = mixed_policy_target(
            batch["p"],
            incumbent_policy_batch(parent_model, batch),
            batch["mask"],
            pr258.BETA,
        )
        policy_loss, value_loss = _losses(model, {**batch, "p": target})
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        if step in TELEMETRY_STEPS:
            snapshots[step] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            optimizers[step] = copy.deepcopy(optimizer.state_dict())
    if replay.optimizer_state_sha256(adam) != pr258.ADAM_SHA:
        fail("pristine optimizer changed")
    return snapshots, optimizers


def validation_metrics_by_replay(
    snapshots: dict[int, dict[str, torch.Tensor]],
    lane: str,
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    parent: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
) -> dict[str, Any]:
    rows = (
        {"a": rows_a, "b": rows_b}
        if lane == "pair_mix"
        else {"single": rows_a if lane == "single_a" else rows_b}
    )
    parent_policies = {key: pr258.policy(parent, value) for key, value in rows.items()}
    result: dict[str, Any] = {}
    for checkpoint, state in {
        "initial": initial,
        "16": snapshots[16],
        "32": snapshots[32],
    }.items():
        by_replay = {
            key: pr258.metric_row(state, value, parent_policies[key], initial)[
                "ce_beta095"
            ]
            for key, value in rows.items()
        }
        result[checkpoint] = float(np.mean(list(by_replay.values())))
        result[f"{checkpoint}_by_replay"] = by_replay
    result["validation_change_16_32"] = result["16"] - result["32"]
    return result


def geometry(
    snapshots: dict[int, dict[str, torch.Tensor]],
    initial: dict[str, torch.Tensor],
    gradient: torch.Tensor | None = None,
) -> dict[str, float]:
    d16, d32 = (
        pr258.adapter_vector(snapshots[16], initial),
        pr258.adapter_vector(snapshots[32], initial),
    )
    epoch2 = d32 - d16
    result = {
        "norm_d16": float(torch.linalg.vector_norm(d16)),
        "norm_d32": float(torch.linalg.vector_norm(d32)),
        "norm_second_epoch": float(torch.linalg.vector_norm(epoch2)),
        "cosine_d16_d32": pr258.cosine(d16, d32),
        "cosine_second_epoch_d16": pr258.cosine(epoch2, d16),
    }
    if gradient is not None:
        result["cosine_second_epoch_negative_full_replay_mean_gradient"] = pr258.cosine(
            epoch2, -gradient
        )
    return result


def per_opening(effect: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            effect["per_opening_effect"][key]
            for key in sorted(effect["per_opening_effect"])
        ]
    )


def analyses(evaluation: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {"per_pair": {}}
    bootstrap_input: dict[str, list[list[dict[str, np.ndarray]]]] = {
        "delta_16": [],
        "delta_32": [],
        "interaction": [],
        "mix_duration": [],
        "single_duration": [],
    }
    for pair_index in range(1, 6):
        pair = f"pair{pair_index}"
        suite_rows, strengths = (
            [],
            {
                step: {lane: [] for lane in ("single_a", "single_b", "pair_mix")}
                for step in CHECKPOINTS
            },
        )
        for suite in SUITE_SEEDS:
            values = evaluation[suite][pair]
            delta = {}
            for step in CHECKPOINTS:
                mix, a, b = (
                    per_opening(values[str(step)][lane]["effect"])
                    for lane in ("pair_mix", "single_a", "single_b")
                )
                delta[step] = mix - 0.5 * (a + b)
                for lane in strengths[step]:
                    strengths[step][lane].append(
                        values[str(step)][lane]["effect"]["paired_candidate_effect"]
                    )
            mix_duration = per_opening(
                values["32"]["pair_mix"]["effect"]
            ) - per_opening(values["16"]["pair_mix"]["effect"])
            single_duration = 0.5 * (
                (
                    per_opening(values["32"]["single_a"]["effect"])
                    - per_opening(values["16"]["single_a"]["effect"])
                )
                + (
                    per_opening(values["32"]["single_b"]["effect"])
                    - per_opening(values["16"]["single_b"]["effect"])
                )
            )
            suite_rows.append(
                {
                    "delta_16": delta[16],
                    "delta_32": delta[32],
                    "interaction": delta[32] - delta[16],
                    "mix_duration": mix_duration,
                    "single_duration": single_duration,
                }
            )
        bootstrap_input["delta_16"].append(suite_rows)
        bootstrap_input["delta_32"].append(suite_rows)
        bootstrap_input["interaction"].append(suite_rows)
        bootstrap_input["mix_duration"].append(suite_rows)
        bootstrap_input["single_duration"].append(suite_rows)
        single_mean = {
            str(step): 0.5
            * (
                np.mean(strengths[step]["single_a"])
                + np.mean(strengths[step]["single_b"])
            )
            for step in CHECKPOINTS
        }
        mix_mean = {
            str(step): float(np.mean(strengths[step]["pair_mix"]))
            for step in CHECKPOINTS
        }
        data["per_pair"][pair] = {
            "delta_16": float(
                np.concatenate([row["delta_16"] for row in suite_rows]).mean()
            ),
            "delta_32": float(
                np.concatenate([row["delta_32"] for row in suite_rows]).mean()
            ),
            "interaction": float(
                np.concatenate([row["interaction"] for row in suite_rows]).mean()
            ),
            "suite_deltas": {
                suite: {
                    "16": float(row["delta_16"].mean()),
                    "32": float(row["delta_32"].mean()),
                }
                for suite, row in zip(SUITE_SEEDS, suite_rows, strict=True)
            },
            "single_mean_effect": single_mean,
            "pair_mix_effect": mix_mean,
            "mix_duration": float(
                np.concatenate([row["mix_duration"] for row in suite_rows]).mean()
            ),
            "single_duration": float(
                np.concatenate([row["single_duration"] for row in suite_rows]).mean()
            ),
        }
    rng = np.random.default_rng(42)

    def bootstrap(metric: str) -> dict[str, float]:
        draws = []
        for _ in range(10_000):
            values = []
            for pair_index in rng.integers(0, 5, 5):
                row = bootstrap_input[metric][pair_index][rng.integers(0, 3)]
                sample = row[metric]
                values.append(
                    float(rng.choice(sample, len(sample), replace=True).mean())
                )
            draws.append(float(np.mean(values)))
        return {
            "mean": float(np.mean(draws)),
            "lower_95": float(np.quantile(draws, 0.025)),
            "upper_95": float(np.quantile(draws, 0.975)),
        }

    for metric in bootstrap_input:
        data[metric] = bootstrap(metric)
    for metric in ("delta_16", "delta_32", "interaction"):
        values = np.asarray([data["per_pair"][f"pair{i}"][metric] for i in range(1, 6)])
        data[metric].update(
            {
                "observed_mean": float(values.mean()),
                "median": float(np.median(values)),
                "sd": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
                "positive_pairs": int((values > 0).sum()),
            }
        )
    data["absolute_strength_variance"] = {
        "single_mean_step16": float(
            np.var(
                [
                    data["per_pair"][f"pair{i}"]["single_mean_effect"]["16"]
                    for i in range(1, 6)
                ],
                ddof=1,
            )
        ),
        "single_mean_step32": float(
            np.var(
                [
                    data["per_pair"][f"pair{i}"]["single_mean_effect"]["32"]
                    for i in range(1, 6)
                ],
                ddof=1,
            )
        ),
        "pair_mix_step16": float(
            np.var(
                [
                    data["per_pair"][f"pair{i}"]["pair_mix_effect"]["16"]
                    for i in range(1, 6)
                ],
                ddof=1,
            )
        ),
        "pair_mix_step32": float(
            np.var(
                [
                    data["per_pair"][f"pair{i}"]["pair_mix_effect"]["32"]
                    for i in range(1, 6)
                ],
                ddof=1,
            )
        ),
    }
    return data


def classify(
    primary: dict[str, Any], training: dict[str, Any], shallow: dict[str, Any]
) -> str:
    delta32, interaction = primary["delta_32"], primary["interaction"]
    values = [primary["per_pair"][f"pair{i}"]["delta_32"] for i in range(1, 6)]
    ce_improved = all(
        lane["metrics"]["improvement_0_32"] > 0
        for pair in training.values()
        for lane in pair.values()
    )
    success = (
        delta32["observed_mean"] > 0
        and delta32["lower_95"] > 0
        and delta32["positive_pairs"] >= 4
        and min(values) >= -0.02
        and interaction["observed_mean"] > 0
        and interaction["lower_95"] > 0
        and ce_improved
    )
    shallow_values = [shallow["per_pair"][f"pair{i}"]["delta_32"] for i in range(1, 6)]
    shallow_harm = (
        shallow["delta_32"]["observed_mean"] < -0.02
        or sum(value < -0.02 for value in shallow_values) >= 2
    )
    if success:
        return (
            "aggregation_longer_training_causes_shallow_harm"
            if shallow_harm
            else "second_epoch_unlocks_replay_aggregation"
        )
    mix_duration, single_duration = (
        primary["mix_duration"]["mean"],
        primary["single_duration"]["mean"],
    )
    validation_changes = [
        lane["validation"]["validation_change_16_32"]
        for pair in training.values()
        for lane in pair.values()
    ]
    if (
        ce_improved
        and all(change < 0 for change in validation_changes)
        and mix_duration <= 0
    ):
        return "second_epoch_overfits"
    if mix_duration > 0 and single_duration > 0 and interaction["lower_95"] <= 0:
        return "longer_training_is_generic_not_aggregation_specific"
    if ce_improved and mix_duration <= 0:
        return "second_epoch_reduces_underfit_but_not_strength"
    return "aggregation_still_not_helpful"


def evaluate_frozen(workdir: Path) -> None:
    """Evaluate only already-frozen models and already-consumed Y/Z/AA suites."""
    frozen = load_json(workdir / "frozen_manifest.json")
    registry = registry_module.load(workdir)
    expected_suites = {
        label: spec.sha256 for label, spec in registry.items() if label in SUITE_SEEDS
    }
    actual_suites = {
        label: sha256_file(workdir / "suites" / f"suite_{label}.jsonl")
        for label in SUITE_SEEDS
    }
    if actual_suites != expected_suites:
        fail("sealed Y/Z/AA suite mismatch")
    flat = {}
    for pair, lanes in frozen["training"].items():
        for lane, details in lanes.items():
            for step in CHECKPOINTS:
                checkpoint = workdir / f"train/{pair}/{lane}/step_{step:04d}.pt"
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                expected = details["checkpoints"][str(step)]
                if (
                    contract.state_sha256(saved["model"]) != expected["model_sha256"]
                    or replay.optimizer_state_sha256(saved["optimizer"])
                    != expected["optimizer_sha256"]
                ):
                    fail(f"frozen model mismatch for {pair} {lane} step{step}")
                flat[f"{pair}_{step}_{lane}"] = checkpoint.with_suffix("") / "artifact"
    paths = {
        label: workdir / "suites" / f"suite_{label}.jsonl" for label in SUITE_SEEDS
    }

    def evaluate(context: str) -> dict[str, Any]:
        raw = pr258.evaluate(flat, paths, workdir, context)
        return {
            suite: {
                pair: {
                    str(step): {
                        lane: raw[suite][f"{pair}_{step}_{lane}"]
                        for lane in ("single_a", "single_b", "pair_mix")
                    }
                    for step in CHECKPOINTS
                }
                for pair in frozen["training"]
            }
            for suite in SUITE_SEEDS
        }

    primary = analyses(evaluate("1200:1200"))
    (workdir / "primary_1200_results.json").write_text(
        json.dumps(primary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shallow = analyses(evaluate("384:256"))
    result = {
        **frozen,
        "primary_evaluation": primary,
        "shallow_evaluation": shallow,
        "classification": classify(primary, frozen["training"], shallow),
    }
    (workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr259_two_replay_second_epoch"),
    )
    parser.add_argument("--evaluate-frozen", action="store_true")
    args = parser.parse_args()
    if args.evaluate_frozen:
        evaluate_frozen(args.workdir)
        return
    started = time.monotonic()
    registry = registry_module.load(args.workdir)
    replays, frozen_plans, source_candidates = verify_source(registry)
    snapshot = torch.load(replay.A16_SNAPSHOT, map_location="cpu", weights_only=False)
    a16, adam = replay.immutable_initial_state(snapshot)
    if (
        sha256_file(replay.A16_SNAPSHOT) != pr258.A16_SHA
        or replay.optimizer_state_sha256(adam) != pr258.ADAM_SHA
    ):
        fail("A16 or Adam mismatch")
    p1 = new_model(torch.device("cpu"))
    pr258.load_checkpoint_into_model(p1, replay.P1_CHECKPOINT)
    parent = {
        key: value.detach().cpu().clone() for key, value in p1.state_dict().items()
    }
    lanes = lanes_from_manifest(replays, frozen_plans)
    training, candidates, geometry_report = {}, {}, {}
    for pair_index, (seed_a, seed_b) in enumerate(pr258.PAIRS, 1):
        pair = f"pair{pair_index}"
        training[pair], candidates[pair], geometry_report[pair] = {}, {}, {}
        gradient = 0.5 * (
            pr258.full_gradient(replays[seed_a], a16, parent)
            + pr258.full_gradient(replays[seed_b], a16, parent)
        )
        validation_a = lanes[pair]["single_a"][2]
        validation_b = lanes[pair]["single_b"][2]
        for lane, (rows, batches, _validation) in lanes[pair].items():
            snapshots, optimizers = train_twice(rows, batches, a16, adam, parent)
            expected = source_candidates["training"][pair][lane]["checkpoints"]["16"]
            saved = torch.load(
                SOURCE / f"train/{pair}/{lane}/step_0016.pt",
                map_location="cpu",
                weights_only=False,
            )
            if (
                contract.state_sha256(snapshots[16]) != expected["model_sha256"]
                or replay.optimizer_state_sha256(optimizers[16])
                != expected["optimizer_sha256"]
                or not state_equal(snapshots[16], saved["model"])
                or replay.optimizer_state_sha256(saved["optimizer"])
                != expected["optimizer_sha256"]
                or not adapter_equal(snapshots[16], saved["model"])
            ):
                fail(f"step16 reproduction mismatch for {pair} {lane}")
            metric_rows = (
                rows if lane != "pair_mix" else replays[seed_a] + replays[seed_b]
            )
            telemetry = pr258.metrics(snapshots, metric_rows, parent, a16)
            telemetry["improvement_0_16"] = (
                telemetry["initial"]["ce_beta095"] - telemetry["16"]["ce_beta095"]
            )
            telemetry["improvement_16_32"] = (
                telemetry["16"]["ce_beta095"] - telemetry["32"]["ce_beta095"]
            )
            telemetry["improvement_0_32"] = (
                telemetry["initial"]["ce_beta095"] - telemetry["32"]["ce_beta095"]
            )
            training[pair][lane] = {
                "plan": frozen_plans["sample_plans"][pair]["plans"][lane],
                "metrics": telemetry,
                "validation": validation_metrics_by_replay(
                    snapshots, lane, validation_a, validation_b, parent, a16
                ),
                "checkpoints": {
                    str(step): {
                        "model_sha256": contract.state_sha256(snapshots[step]),
                        "optimizer_sha256": replay.optimizer_state_sha256(
                            optimizers[step]
                        ),
                    }
                    for step in CHECKPOINTS
                },
            }
            geometry_report[pair][lane] = geometry(
                snapshots, a16, gradient if lane == "pair_mix" else None
            )
            for step in CHECKPOINTS:
                checkpoint = args.workdir / f"train/{pair}/{lane}/step_{step:04d}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": snapshots[step], "optimizer": optimizers[step]},
                    checkpoint,
                )
                candidates[pair][str(step), lane] = export(
                    snapshots[step],
                    checkpoint.with_suffix(""),
                    f"pr259_{pair}_{lane}_step{step}",
                )
        if pair_index == 1:
            repeated, repeated_adam = train_twice(
                *lanes[pair]["pair_mix"][:2], a16, adam, parent
            )
            if (
                contract.state_sha256(repeated[32])
                != training[pair]["pair_mix"]["checkpoints"]["32"]["model_sha256"]
                or replay.optimizer_state_sha256(repeated_adam[32])
                != training[pair]["pair_mix"]["checkpoints"]["32"]["optimizer_sha256"]
            ):
                fail("repeated step32 pair_mix mismatch")
    frozen = {
        "schema": "alphazero_lite_pr259_two_replay_second_epoch_v1",
        "source_pr258_manifest_sha256": sha256_file(SOURCE / "frozen_manifest.json"),
        "training": training,
        "geometry": geometry_report,
        "compute": {
            "new_self_play_games": 0,
            "unique_examples_per_candidate": 8192,
            "step16_exposures": 8192,
            "step32_exposures": 16384,
            "optimizer_steps": 32,
        },
    }
    frozen["candidate_model_sha256"] = pr258.canonical_sha(
        {
            pair: {
                f"{step}_{lane}": training[pair][lane]["checkpoints"][str(step)][
                    "model_sha256"
                ]
                for step in CHECKPOINTS
                for lane in ("single_a", "single_b", "pair_mix")
            }
            for pair in training
        }
    )
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "frozen_candidates.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pr258.SUITE_SEEDS = SUITE_SEEDS
    paths, suite_manifest, preflight = pr258.seal_suites(
        args.workdir, registry, replays
    )
    if not preflight["passed"]:
        fail("suite preflight failed")
    frozen["suite_manifest"], frozen["preflight"] = suite_manifest, preflight
    (args.workdir / "frozen_manifest.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat = {
        f"{pair}_{step}_{lane}": path
        for pair, values in candidates.items()
        for (step, lane), path in values.items()
    }

    def evaluate(context: str) -> dict[str, Any]:
        raw = pr258.evaluate(flat, paths, args.workdir, context)
        return {
            suite: {
                pair: {
                    str(step): {
                        lane: raw[suite][f"{pair}_{step}_{lane}"]
                        for lane in ("single_a", "single_b", "pair_mix")
                    }
                    for step in CHECKPOINTS
                }
                for pair in training
            }
            for suite in SUITE_SEEDS
        }

    primary = analyses(evaluate("1200:1200"))
    (args.workdir / "primary_1200_results.json").write_text(
        json.dumps(primary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shallow = analyses(evaluate("384:256"))
    result = {
        **frozen,
        "primary_evaluation": primary,
        "shallow_evaluation": shallow,
        "classification": classify(primary, training, shallow),
        "wall_clock_seconds": time.monotonic() - started,
    }
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
