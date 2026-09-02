#!/usr/bin/env python3
# ruff: noqa: E402
"""Matched 582-parameter trunk-adapter versus policy-readout experiment.

This runner is deliberately fail-closed: it reuses PR258's committed single
replay plans, starts every lane at A16, and seals every candidate before AE/AF/AG
are generated. It never generates self-play or updates a value/trunk tensor.
"""

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
from ml.alphazero_lite import run_pr258_two_replay_aggregation as pr258
from ml.alphazero_lite import run_pr259_two_replay_second_epoch as pr259
from ml.alphazero_lite import run_pr260_value_head_refresh as pr260
from ml.alphazero_lite.evaluation_metrics import paired_effect_difference
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import sha256_file
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    export,
    new_model,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (
    incumbent_policy_batch,
    mixed_policy_target,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch, _losses
from ml.alphazero_lite.run_shared_trunk_delta_attribution import js
from ml.alphazero_lite.train import apply_trainable_scope

SEEDS = tuple(range(53, 63))
STEPS = (1, 4, 16)
BATCH_SIZE, BETA, LR = 512, 0.95, 1e-5
LANES = {
    "trunk_adapter": "policy_adapter_only",
    "policy_readout": "policy_readout_only",
}
KEYS = {
    "trunk_adapter": ("policy_adapter.weight", "policy_adapter.bias"),
    "policy_readout": ("policy_head.weight", "policy_head.bias"),
}
PARAMETER_COUNTS = {
    "trunk_adapter": 582,
    "policy_readout": 582,
}
SUITE_SEEDS = {"AE": 31042, "AF": 32042, "AG": 33042}


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def parameters(model: torch.nn.Module, lane: str) -> list[torch.nn.Parameter]:
    named = dict(model.named_parameters())
    result = [named[key] for key in KEYS[lane]]
    if sum(parameter.numel() for parameter in result) != PARAMETER_COUNTS[lane]:
        fail(f"{lane} trainable parameter count is not {PARAMETER_COUNTS[lane]}")
    if {name for name, parameter in named.items() if parameter.requires_grad} != set(
        KEYS[lane]
    ):
        fail(f"{lane} scope trains an unexpected tensor")
    return result


def probabilities(
    state: dict[str, torch.Tensor], rows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    return pr260.policy_logits_and_probabilities(state, rows)


def check_identity(
    a16: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    lane: str,
    rows: list[dict[str, Any]],
) -> None:
    changed = set(KEYS[lane])
    if any(not torch.equal(a16[key], state[key]) for key in a16 if key not in changed):
        fail(f"{lane} changed frozen trunk, value, or opposing policy tensors")
    if lane == "trunk_adapter" and any(
        not torch.equal(a16[key], state[key]) for key in KEYS["policy_readout"]
    ):
        fail("trunk_adapter policy_head changed")
    if lane == "policy_readout" and any(
        not torch.equal(a16[key], state[key]) for key in KEYS["trunk_adapter"]
    ):
        fail("policy_readout policy_adapter changed")
    before, after = (
        pr260.value_predictions(a16, rows),
        pr260.value_predictions(state, rows),
    )
    if not np.array_equal(before, after):
        fail(f"{lane} value output changed")


def metric(
    state: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    p1: np.ndarray,
    lane: str,
) -> dict[str, Any]:
    logits, policy = probabilities(state, rows)
    initial_logits, initial_policy = probabilities(initial, rows)
    search = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    beta = 0.05 * search + 0.95 * p1
    vector = torch.cat([(state[key] - initial[key]).reshape(-1) for key in KEYS[lane]])
    return {
        "ce_beta095": float(np.mean(pr258._cross_entropy(policy, beta))),
        "ce_raw_reused_target": float(np.mean(pr258._cross_entropy(policy, search))),
        "ce_p1": float(np.mean(pr258._cross_entropy(policy, p1))),
        "legal_policy_l1_vs_a16": float(
            np.abs(policy - initial_policy).sum(axis=1).mean()
        ),
        "legal_policy_l1_vs_p1": float(np.abs(policy - p1).sum(axis=1).mean()),
        "js_vs_p1": float(js(policy, p1).mean()),
        "top1_disagreement_vs_a16": float(
            np.mean(np.argmax(policy, axis=1) != np.argmax(initial_policy, axis=1))
        ),
        "parameter_delta_norm": float(torch.linalg.vector_norm(vector)),
        "logit_delta_rms": float(np.sqrt(np.mean((logits - initial_logits) ** 2))),
        "model_sha256": pr258.contract.state_sha256(state),
    }


def train_lane(
    rows: list[dict[str, Any]],
    batches: list[list[int]],
    a16: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
    lane: str,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]], str]:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(copy.deepcopy(a16))
    apply_trainable_scope(model, LANES[lane])
    trainable = parameters(model, lane)
    optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=0.0)
    initial_optimizer_sha = replay.optimizer_state_sha256(optimizer.state_dict())
    parent_model = new_model(torch.device("cpu"))
    parent_model.load_state_dict(parent)
    for parameter in parent_model.parameters():
        parameter.requires_grad_(False)
    snapshots, optimizers = (
        {0: state_copy(model)},
        {0: copy.deepcopy(optimizer.state_dict())},
    )
    model.train()
    for step, indexes in enumerate(batches, 1):
        batch = _batch(rows, np.asarray(indexes, dtype=np.int64), torch.device("cpu"))
        target = mixed_policy_target(
            batch["p"], incumbent_policy_batch(parent_model, batch), batch["mask"], BETA
        )
        policy_loss, value_loss = _losses(model, {**batch, "p": target})
        optimizer.zero_grad(set_to_none=True)
        policy_loss.backward(retain_graph=True)
        policy_grads = [parameter.grad.detach().clone() for parameter in trainable]
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        if any(
            not torch.equal(before, parameter.grad)
            for before, parameter in zip(policy_grads, trainable, strict=True)
        ):
            fail(f"{lane} value loss contributed policy gradient")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step in STEPS:
            snapshots[step], optimizers[step] = (
                state_copy(model),
                copy.deepcopy(optimizer.state_dict()),
            )
    return snapshots, optimizers, initial_optimizer_sha


def comparison(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    a_logits, a = probabilities(left, rows)
    b_logits, b = probabilities(right, rows)
    initial_logits, _ = probabilities(initial, rows)
    da, db = a_logits - initial_logits, b_logits - initial_logits
    cosine = np.sum(da * db, axis=1) / (
        np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1)
    )
    top_two = np.argpartition(initial_logits, -2, axis=1)[:, -2:]
    margins = (
        np.take_along_axis(a_logits, top_two, 1)[:, 0]
        - np.take_along_axis(a_logits, top_two, 1)[:, 1]
    )
    other = (
        np.take_along_axis(b_logits, top_two, 1)[:, 0]
        - np.take_along_axis(b_logits, top_two, 1)[:, 1]
    )
    base = np.take_along_axis(initial_logits, top_two, 1)
    base_margin = base[:, 0] - base[:, 1]
    return {
        "mean_policy_l1": float(np.abs(a - b).sum(axis=1).mean()),
        "js": float(js(a, b).mean()),
        "top1_disagreement": float(
            np.mean(np.argmax(a, axis=1) != np.argmax(b, axis=1))
        ),
        "logit_delta_cosine_relative_a16": float(np.nanmean(cosine)),
        "opposite_top_two_margin_fraction": float(
            np.mean((margins - base_margin) * (other - base_margin) < 0)
        ),
    }


def analyze(
    raw: dict[str, Any], labels: tuple[str, ...], challenger: str = "policy_readout"
) -> dict[str, Any]:
    per_seed, nested = {}, []
    for seed in SEEDS:
        suites, readout_strength, adapter_strength = {}, [], []
        samples = []
        for label in labels:
            values = raw[label]
            diff = paired_effect_difference(
                values[f"seed{seed}_{challenger}"]["effect"],
                values[f"seed{seed}_trunk_adapter"]["effect"],
            )
            vector = np.asarray(
                [
                    diff["per_opening_effect"][key]
                    for key in sorted(diff["per_opening_effect"])
                ]
            )
            suites[label], samples = float(vector.mean()), [*samples, vector]
            readout_strength.append(
                values[f"seed{seed}_{challenger}"]["effect"]["paired_candidate_effect"]
            )
            adapter_strength.append(
                values[f"seed{seed}_trunk_adapter"]["effect"]["paired_candidate_effect"]
            )
        delta = float(np.concatenate(samples).mean())
        per_seed[str(seed)] = {
            "suite_deltas": suites,
            "delta": delta,
            "opening_bootstrap_ci": [
                float(
                    np.quantile(
                        [
                            np.concatenate(samples)[
                                np.random.default_rng(i).integers(
                                    0,
                                    len(np.concatenate(samples)),
                                    len(np.concatenate(samples)),
                                )
                            ].mean()
                            for i in range(1000)
                        ],
                        q,
                    )
                )
                for q in (0.025, 0.975)
            ],
            f"{challenger}_absolute_adjusted_p1_strength": float(
                np.mean(readout_strength)
            ),
            "trunk_adapter_absolute_adjusted_p1_strength": float(
                np.mean(adapter_strength)
            ),
        }
        nested.append(samples)
    values, rng, draws = (
        np.asarray([per_seed[str(seed)]["delta"] for seed in SEEDS]),
        np.random.default_rng(42),
        [],
    )
    for _ in range(10_000):
        draws.append(
            float(
                np.mean(
                    [
                        rng.choice(
                            nested[index][rng.integers(0, len(labels))],
                            128,
                            replace=True,
                        ).mean()
                        for index in rng.integers(0, len(SEEDS), len(SEEDS))
                    ]
                )
            )
        )
    return {
        "per_seed": per_seed,
        "primary": {
            "mean_delta": float(values.mean()),
            "median": float(np.median(values)),
            "sd": float(values.std(ddof=1)),
            "range": [float(values.min()), float(values.max())],
            "positive_seed_count": int((values > 0).sum()),
            "hierarchical_replay_seed_suite_opening_ci": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        },
    }


def classify(
    primary: dict[str, Any], learned: int, shallow: dict[str, Any] | None
) -> str:
    values, summary = (
        [value["delta"] for value in primary["per_seed"].values()],
        primary["primary"],
    )
    if learned < 8:
        return "policy_readout_does_not_learn"
    success = (
        summary["mean_delta"] > 0
        and summary["hierarchical_replay_seed_suite_opening_ci"][0] > 0
        and summary["positive_seed_count"] >= 8
        and min(values) >= -0.02
    )
    shallow_harm = shallow is not None and (
        shallow["primary"]["mean_delta"] < -0.02
        or sum(value["delta"] < -0.02 for value in shallow["per_seed"].values()) >= 3
    )
    if success:
        return (
            "policy_readout_improves_high_budget_with_shallow_harm"
            if shallow_harm
            else "policy_readout_basis_improves_strength"
        )
    if summary["mean_delta"] < -0.02:
        return "trunk_adapter_remains_better"
    if summary["mean_delta"] > 0:
        return "policy_readout_is_seed_sensitive"
    return "policy_basis_changes_fit_not_strength"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr261_policy_representation")
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    args, started = parser.parse_args(), time.monotonic()
    registry = registry_module.load(args.workdir)
    through_ad = [
        "canonical",
        *"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "AA",
        "AB",
        "AC",
        "AD",
    ]
    if list(registry) != through_ad:
        fail("AE/AF/AG are already consumed or registry is not canonical through AD")
    replays, frozen_plans, _source_candidates = pr260.load_source(registry)
    snapshot = torch.load(replay.A16_SNAPSHOT, map_location="cpu", weights_only=False)
    a16, _old_adam = replay.immutable_initial_state(snapshot)
    if sha256_file(replay.A16_SNAPSHOT) != pr258.A16_SHA:
        fail("A16 model hash mismatch")
    parent_model = new_model(torch.device("cpu"))
    pr258.load_checkpoint_into_model(parent_model, replay.P1_CHECKPOINT)
    parent = state_copy(parent_model)
    lanes, training, artifacts, candidates = (
        pr259.lanes_from_manifest(replays, frozen_plans),
        {},
        {},
        {},
    )
    all_rows = [row for rows in replays.values() for row in rows]
    for pair, pair_lanes in lanes.items():
        for lane_name, seed in (
            ("single_a", int(pair[-1]) * 2 + 51),
            ("single_b", int(pair[-1]) * 2 + 52),
        ):
            rows, batches, validation = pair_lanes[lane_name]
            states, optimizers, initial_optimizer_sha = {}, {}, None
            for representation in LANES:
                snapshots, optimizer_states, optimizer_sha = train_lane(
                    rows, batches, a16, parent, representation
                )
                initial_optimizer_sha = initial_optimizer_sha or optimizer_sha
                if optimizer_sha != initial_optimizer_sha:
                    fail("fresh Adam initial state differs by lane")
                check_identity(a16, snapshots[0], representation, all_rows)
                initial_logits, initial_policy = probabilities(a16, rows)
                lane_logits, lane_policy = probabilities(snapshots[0], rows)
                if not np.array_equal(
                    initial_logits, lane_logits
                ) or not np.array_equal(initial_policy, lane_policy):
                    fail("initial policy mismatch")
                check_identity(a16, snapshots[16], representation, all_rows)
                states[representation], optimizers[representation] = (
                    snapshots,
                    optimizer_states,
                )
            p1_train, p1_validation = (
                probabilities(parent, rows)[1],
                probabilities(parent, validation)[1],
            )
            training[str(seed)] = {
                "plan": frozen_plans["sample_plans"][pair]["plans"][lane_name],
                "initial_optimizer_sha256": initial_optimizer_sha,
                "lanes": {
                    representation: {
                        "trainable_parameters": KEYS[representation],
                        "trainable_parameter_count": 582,
                        "metrics": {
                            str(step) if step else "initial": metric(
                                state, a16, rows, p1_train, representation
                            )
                            | {
                                "optimizer_sha256": replay.optimizer_state_sha256(
                                    optimizers[representation][step]
                                )
                            }
                            for step, state in states[representation].items()
                        },
                        "heldout_beta095_ce": {
                            "initial": metric(
                                a16, a16, validation, p1_validation, representation
                            )["ce_beta095"],
                            "16": metric(
                                states[representation][16],
                                a16,
                                validation,
                                p1_validation,
                                representation,
                            )["ce_beta095"],
                        },
                    }
                    for representation in LANES
                },
                "output_space_step16": comparison(
                    states["trunk_adapter"][16], states["policy_readout"][16], a16, rows
                ),
            }
            if seed in (53, 60):
                reproducibility = {}
                for representation in reversed(tuple(LANES)):
                    repeated_states, repeated_optimizers, repeated_initial_sha = (
                        train_lane(rows, batches, a16, parent, representation)
                    )
                    expected_model = pr258.contract.state_sha256(
                        states[representation][16]
                    )
                    expected_optimizer = replay.optimizer_state_sha256(
                        optimizers[representation][16]
                    )
                    actual_model = pr258.contract.state_sha256(repeated_states[16])
                    actual_optimizer = replay.optimizer_state_sha256(
                        repeated_optimizers[16]
                    )
                    if (
                        repeated_initial_sha != initial_optimizer_sha
                        or actual_model != expected_model
                        or actual_optimizer != expected_optimizer
                    ):
                        fail(f"seed {seed} repeated/reverse-order lane mismatch")
                    reproducibility[representation] = {
                        "step16_model_sha256": actual_model,
                        "step16_optimizer_sha256": actual_optimizer,
                    }
                training[str(seed)]["repeated_lane_reverse_order"] = reproducibility
            directory = args.workdir / "train" / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            for representation in LANES:
                torch.save(
                    {
                        "model": states[representation][16],
                        "optimizer": optimizers[representation][16],
                    },
                    directory / f"{representation}_step16.pt",
                )
                artifacts.setdefault(str(seed), {})[representation] = export(
                    states[representation][16],
                    directory / representation,
                    f"pr261_seed{seed}_{representation}",
                )
                candidates[f"seed{seed}_{representation}"] = artifacts[str(seed)][
                    representation
                ]
    frozen = {
        "schema": "alphazero_lite_pr261_policy_representation_v1",
        "a16_model_sha256": pr258.A16_SHA,
        "replays": frozen_plans["replays"],
        "training": training,
        "candidate_model_sha256": pr258.canonical_sha(
            {
                f"seed{seed}_{representation}": training[str(seed)]["lanes"][
                    representation
                ]["metrics"]["16"]["model_sha256"]
                for seed in SEEDS
                for representation in LANES
            }
        ),
    }
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "frozen_candidates.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pr258.SUITE_SEEDS = SUITE_SEEDS
    suite_paths, suite_manifest, preflight = pr258.seal_suites(
        args.workdir, registry, replays
    )
    if not preflight["passed"]:
        fail("AE/AF/AG preflight failed")
    frozen |= {
        "suite_manifest": suite_manifest,
        "preflight": preflight,
        "suite_manifest_sha256": pr258.canonical_sha(suite_manifest),
        "preflight_sha256": pr258.canonical_sha(preflight),
    }
    (args.workdir / "frozen_manifest.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.freeze_only:
        return
    pr258.WORKERS = args.workers
    primary = analyze(
        pr258.evaluate(candidates, suite_paths, args.workdir, "1200:1200"),
        tuple(SUITE_SEEDS),
    )
    (args.workdir / "primary_1200_results.json").write_text(
        json.dumps(primary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shallow = analyze(
        pr258.evaluate(candidates, suite_paths, args.workdir, "384:256"),
        tuple(SUITE_SEEDS),
    )
    learned = sum(
        value["lanes"]["policy_readout"]["heldout_beta095_ce"]["16"]
        < value["lanes"]["policy_readout"]["heldout_beta095_ce"]["initial"]
        for value in training.values()
    )
    result = frozen | {
        "primary_evaluation": primary,
        "shallow_evaluation": shallow,
        "heldout_beta095_ce_improved_seeds": learned,
        "classification": classify(primary, learned, shallow),
        "wall_clock_seconds": time.monotonic() - started,
    }
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
