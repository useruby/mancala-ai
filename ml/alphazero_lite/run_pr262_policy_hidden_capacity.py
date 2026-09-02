#!/usr/bin/env python3
# ruff: noqa: E402
"""Ten-replay-seed hidden-policy capacity test against frozen PR #261 adapters."""

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
from ml.alphazero_lite import run_pr260_value_head_refresh as pr260
from ml.alphazero_lite import run_pr261_policy_representation as pr261
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
from ml.alphazero_lite.train import apply_trainable_scope

SOURCE = Path("/tmp/azlite_pr261_policy_representation")
SEEDS, STEPS = tuple(range(53, 63)), (1, 4, 16)
KEYS = ("policy_hidden_layer.weight", "policy_hidden_layer.bias")
SUITE_SEEDS = {"AH": 34042, "AI": 35042, "AJ": 36042}


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def train_hidden(
    rows: list[dict[str, Any]],
    batches: list[list[int]],
    a16: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
) -> tuple[
    dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]], dict[int, float]
]:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(copy.deepcopy(a16))
    apply_trainable_scope(model, "policy_hidden_only")
    named = dict(model.named_parameters())
    trainable = [named[key] for key in KEYS]
    if sum(parameter.numel() for parameter in trainable) != 9312:
        fail("policy_hidden_only trainable parameter count is not 9312")
    if {name for name, parameter in named.items() if parameter.requires_grad} != set(
        KEYS
    ):
        fail("policy_hidden_only trains an unexpected tensor")
    optimizer = torch.optim.Adam(trainable, lr=1e-5, weight_decay=0.0)
    parent_model = new_model(torch.device("cpu"))
    parent_model.load_state_dict(parent)
    for parameter in parent_model.parameters():
        parameter.requires_grad_(False)
    snapshots, optimizers, gradient_norms = (
        {0: state_copy(model)},
        {0: copy.deepcopy(optimizer.state_dict())},
        {0: 0.0},
    )
    model.train()
    for step, indexes in enumerate(batches, 1):
        batch = _batch(rows, np.asarray(indexes, dtype=np.int64), torch.device("cpu"))
        target = mixed_policy_target(
            batch["p"], incumbent_policy_batch(parent_model, batch), batch["mask"], 0.95
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
            fail("value loss contributed hidden-policy gradient")
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step in STEPS:
            snapshots[step] = state_copy(model)
            optimizers[step] = copy.deepcopy(optimizer.state_dict())
            gradient_norms[step] = float(gradient_norm)
    return snapshots, optimizers, gradient_norms


def telemetry(
    state: dict[str, torch.Tensor],
    a16: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    p1: np.ndarray,
    gradient_norm: float,
) -> dict[str, Any]:
    values = pr261.metric(state, a16, rows, p1, "policy_hidden")
    _logits, policy = pr261.probabilities(state, rows)
    _initial_logits, initial_policy = pr261.probabilities(a16, rows)
    update = torch.cat([(state[key] - a16[key]).reshape(-1) for key in KEYS])
    l1 = np.abs(policy - initial_policy).sum(axis=1)
    values |= {
        "hidden_parameter_delta_norm": values.pop("parameter_delta_norm"),
        "gradient_norm": gradient_norm,
        "hidden_update_l2": float(torch.linalg.vector_norm(update)),
        "per_parameter_rms_update": float(torch.sqrt(torch.mean(update.square()))),
        "logit_rms_update": values["logit_delta_rms"],
        "legal_policy_l1_percentiles": {
            str(q): float(np.quantile(l1, q / 100)) for q in (50, 90, 99)
        },
    }
    return values


def load_adapter(seed: int, expected_sha: str) -> tuple[dict[str, torch.Tensor], Path]:
    path = SOURCE / "train" / f"seed{seed}" / "trunk_adapter_step16.pt"
    if not path.is_file():
        fail(f"missing PR261 trunk_adapter candidate for seed {seed}")
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    if pr258.contract.state_sha256(state) != expected_sha:
        fail(f"PR261 trunk_adapter candidate hash mismatch for seed {seed}")
    return (
        {key: value.detach().cpu().clone() for key, value in state.items()},
        path.parent / "trunk_adapter" / "artifact",
    )


def classify(primary: dict[str, Any], learned: int, shallow: dict[str, Any]) -> str:
    values, summary = (
        [v["delta"] for v in primary["per_seed"].values()],
        primary["primary"],
    )
    success = (
        summary["mean_delta"] > 0
        and summary["hierarchical_replay_seed_suite_opening_ci"][0] > 0
        and summary["positive_seed_count"] >= 8
        and min(values) >= -0.02
        and learned >= 8
    )
    shallow_harm = (
        shallow["primary"]["mean_delta"] < -0.02
        or sum(v["delta"] < -0.02 for v in shallow["per_seed"].values()) >= 3
    )
    if success:
        return (
            "policy_hidden_capacity_high_budget_with_shallow_harm"
            if shallow_harm
            else "policy_hidden_capacity_improves_strength"
        )
    if learned < 8:
        return "policy_hidden_does_not_learn"
    if summary["mean_delta"] < -0.02 or sum(value < -0.02 for value in values) >= 3:
        return "policy_hidden_degrades_strength"
    if summary["mean_delta"] > 0:
        return "policy_hidden_capacity_seed_sensitive"
    return "policy_hidden_fit_not_strength"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr262_policy_hidden_capacity")
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    args, started = parser.parse_args(), time.monotonic()
    registry = registry_module.load(args.workdir)
    expected_registry = [
        "canonical",
        *"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "AA",
        "AB",
        "AC",
        "AD",
        "AE",
        "AF",
        "AG",
    ]
    if list(registry) != expected_registry:
        fail("registry is not authoritative through AG")
    replays, frozen_plans, _ = pr260.load_source(registry)
    source = json.loads((SOURCE / "frozen_candidates.json").read_text(encoding="utf-8"))
    snapshot = torch.load(replay.A16_SNAPSHOT, map_location="cpu", weights_only=False)
    a16, _ = replay.immutable_initial_state(snapshot)
    if sha256_file(replay.A16_SNAPSHOT) != pr258.A16_SHA:
        fail("A16 model hash mismatch")
    parent_model = new_model(torch.device("cpu"))
    pr258.load_checkpoint_into_model(parent_model, replay.P1_CHECKPOINT)
    parent, all_rows = (
        state_copy(parent_model),
        [row for rows in replays.values() for row in rows],
    )
    pr261.KEYS["policy_hidden"] = KEYS
    pr261.PARAMETER_COUNTS["policy_hidden"] = 9312
    hidden_states, hidden_artifacts, training, candidates = {}, {}, {}, {}
    for seed in SEEDS:
        pair, lane = f"pair{(seed - 51) // 2}", "single_a" if seed % 2 else "single_b"
        rows, batches, validation = pr260.pr259.lanes_from_manifest(
            replays, frozen_plans
        )[pair][lane]
        snapshots, optimizers, grads = train_hidden(rows, batches, a16, parent)
        pr261.check_identity(a16, snapshots[0], "policy_hidden", all_rows)
        pr261.check_identity(a16, snapshots[16], "policy_hidden", all_rows)
        p1_train, p1_validation = (
            pr261.probabilities(parent, rows)[1],
            pr261.probabilities(parent, validation)[1],
        )
        adapter_state, adapter_artifact = load_adapter(
            seed,
            source["training"][str(seed)]["lanes"]["trunk_adapter"]["metrics"]["16"][
                "model_sha256"
            ],
        )
        directory = args.workdir / "train" / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": snapshots[16], "optimizer": optimizers[16]},
            directory / "policy_hidden_step16.pt",
        )
        artifact = export(
            snapshots[16],
            directory / "policy_hidden",
            f"pr262_seed{seed}_policy_hidden",
        )
        pr261.check_identity(a16, snapshots[16], "policy_hidden", all_rows)
        hidden_states[seed], hidden_artifacts[seed] = snapshots, artifact
        metrics = {
            str(step) if step else "initial": telemetry(
                state, a16, rows, p1_train, grads[step]
            )
            | {"optimizer_sha256": replay.optimizer_state_sha256(optimizers[step])}
            for step, state in snapshots.items()
        }
        training[str(seed)] = {
            "plan": frozen_plans["sample_plans"][pair]["plans"][lane],
            "trainable_parameters": KEYS,
            "trainable_parameter_count": 9312,
            "metrics": metrics,
            "heldout_beta095_ce": {
                "initial": pr261.metric(
                    a16, a16, validation, p1_validation, "policy_hidden"
                )["ce_beta095"],
                "16": pr261.metric(
                    snapshots[16], a16, validation, p1_validation, "policy_hidden"
                )["ce_beta095"],
            },
            "hidden_minus_adapter_ce_improvement": source["training"][str(seed)][
                "lanes"
            ]["trunk_adapter"]["metrics"]["initial"]["ce_beta095"]
            - source["training"][str(seed)]["lanes"]["trunk_adapter"]["metrics"]["16"][
                "ce_beta095"
            ]
            - (metrics["initial"]["ce_beta095"] - metrics["16"]["ce_beta095"]),
        }
        candidates[f"seed{seed}_trunk_adapter"] = adapter_artifact
        candidates[f"seed{seed}_policy_hidden"] = artifact
    for seed in (53, 60):
        pair, lane = f"pair{(seed - 51) // 2}", "single_a" if seed % 2 else "single_b"
        rows, batches, _ = pr260.pr259.lanes_from_manifest(replays, frozen_plans)[pair][
            lane
        ]
        repeated, repeated_optimizers, _ = train_hidden(rows, batches, a16, parent)
        if pr258.contract.state_sha256(repeated[16]) != pr258.contract.state_sha256(
            hidden_states[seed][16]
        ) or replay.optimizer_state_sha256(
            repeated_optimizers[16]
        ) != replay.optimizer_state_sha256(
            torch.load(
                args.workdir / "train" / f"seed{seed}" / "policy_hidden_step16.pt",
                map_location="cpu",
                weights_only=False,
            )["optimizer"]
        ):
            fail(f"seed {seed} repeated hidden-lane mismatch")
        training[str(seed)]["repeated_lane_reverse_order"] = True
    frozen = {
        "schema": "alphazero_lite_pr262_policy_hidden_capacity_v1",
        "a16_model_sha256": pr258.A16_SHA,
        "replays": frozen_plans["replays"],
        "training": training,
        "candidate_model_sha256": pr258.canonical_sha(
            {
                name: pr258.contract.state_sha256(hidden_states[int(name[4:6])][16])
                if name.endswith("policy_hidden")
                else source["training"][name[4:6]]["lanes"]["trunk_adapter"]["metrics"][
                    "16"
                ]["model_sha256"]
                for name in candidates
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
        fail("AH/AI/AJ preflight failed")
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
    probe = [row for seed in SEEDS for row in replays[seed]][:256]
    frozen["frozen_probe_comparison"] = {
        str(seed): {
            context: pr260.search_diagnostics(
                candidates[f"seed{seed}_trunk_adapter"],
                candidates[f"seed{seed}_policy_hidden"],
                probe,
                context,
                args.workers,
            )
            | {"first_divergence_simulation": "not_available"}
            for context in ("384:256", "1200:1200")
        }
        for seed in SEEDS
    }
    primary = pr261.analyze(
        pr260.evaluate(
            candidates, suite_paths, args.workdir, "1200:1200", args.workers
        ),
        tuple(SUITE_SEEDS),
        "policy_hidden",
    )
    shallow = pr261.analyze(
        pr260.evaluate(candidates, suite_paths, args.workdir, "384:256", args.workers),
        tuple(SUITE_SEEDS),
        "policy_hidden",
    )
    learned = sum(
        value["heldout_beta095_ce"]["16"] < value["heldout_beta095_ce"]["initial"]
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
