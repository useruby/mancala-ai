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
EXPECTED_CANDIDATE_SHA = (
    "48c828b98bc8d906d71fb9c7209be3df0889a33096fe14eccddce45ab18a854b"
)
EXPECTED_SUITE_MANIFEST_SHA = (
    "ad077c90dc36deafec19512320e6ddfe9077cba33701c5171615b7f35a68ca9e"
)
EXPECTED_PREFLIGHT_SHA = (
    "6d761626b5b9a17a4efaa6b1f41f806cf4254aabe4ebdc12ce785d216e140747"
)
EXPECTED_SUITE_SHA = {
    "AH": "c16148b43cb652f2dc28ca4b8e94c67f66da471f3cc36d318e73ce3258483784",
    "AI": "1c1de16cc5c4f16696858b054c07747575301e27ff9308f270dfdc4cfd13579b",
    "AJ": "95b3c2dc333a5411562b1a1aeeccb0e093a1af9f7e6f4aa4b61362301416798d",
}


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


def recommended_next_experiment(classification: str) -> str:
    recommendations = {
        "policy_hidden_capacity_improves_strength": (
            "Test a zero-initialized additive nonlinear adapter over frozen "
            "policy_hidden features, preserving inherited P1 tensors."
        ),
        "policy_hidden_capacity_high_budget_with_shallow_harm": (
            "Test the same additive nonlinear adapter concept with a "
            "preregistered shallow-search safety guard."
        ),
        "policy_hidden_capacity_seed_sensitive": (
            "Do not add more head capacity; move to a prospective "
            "multi-generation AlphaZero iteration."
        ),
        "policy_hidden_fit_not_strength": (
            "Close isolated supervised-head fitting; run a true joint "
            "policy/value/trunk AlphaZero generation/replacement cycle."
        ),
        "policy_hidden_degrades_strength": (
            "Keep inherited policy tensors frozen; retain additive adapter "
            "architecture and move to prospective iterative AlphaZero training "
            "rather than further unfreezing."
        ),
    }
    return recommendations.get(
        classification, "No follow-up until the invariant failure is resolved."
    )


def evaluate_only(workdir: Path, workers: int, started: float) -> dict[str, Any]:
    """Complete the frozen experiment without retraining or resealing inputs."""
    frozen_path, manifest_path = (
        workdir / "frozen_candidates.json",
        workdir / "frozen_manifest.json",
    )
    if not frozen_path.is_file() or not manifest_path.is_file():
        fail("evaluate-only requires frozen_candidates.json and frozen_manifest.json")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if frozen["candidate_model_sha256"] != EXPECTED_CANDIDATE_SHA:
        fail("frozen candidate aggregate SHA mismatch")
    if manifest["candidate_model_sha256"] != EXPECTED_CANDIDATE_SHA:
        fail("frozen manifest candidate aggregate SHA mismatch")
    if manifest["suite_manifest_sha256"] != EXPECTED_SUITE_MANIFEST_SHA:
        fail("frozen suite manifest identity mismatch")
    if manifest["preflight_sha256"] != EXPECTED_PREFLIGHT_SHA:
        fail("frozen preflight identity mismatch")
    if (
        pr258.canonical_sha(manifest["suite_manifest"])
        != manifest["suite_manifest_sha256"]
    ):
        fail("frozen suite manifest hash mismatch")
    if pr258.canonical_sha(manifest["preflight"]) != manifest["preflight_sha256"]:
        fail("frozen preflight hash mismatch")

    registry = registry_module.load(workdir)
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
        "AH",
        "AI",
        "AJ",
    ]
    if list(registry) != expected_registry:
        fail("registry is not authoritative through AJ")
    replays, _plans, _source_candidates = pr260.load_source(registry)

    candidates, identities = {}, {}
    source = json.loads((SOURCE / "frozen_candidates.json").read_text(encoding="utf-8"))
    for seed in SEEDS:
        for lane, root, expected in (
            (
                "trunk_adapter",
                SOURCE / "train" / f"seed{seed}",
                source["training"][str(seed)]["lanes"]["trunk_adapter"]["metrics"][
                    "16"
                ]["model_sha256"],
            ),
            (
                "policy_hidden",
                workdir / "train" / f"seed{seed}",
                frozen["training"][str(seed)]["metrics"]["16"]["model_sha256"],
            ),
        ):
            checkpoint = root / f"{lane}_step16.pt"
            artifact = root / lane / "artifact"
            if not checkpoint.is_file() or not (artifact / "model.npz").is_file():
                fail(f"missing frozen {lane} candidate for seed {seed}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)[
                "model"
            ]
            model_sha = pr258.contract.state_sha256(state)
            if model_sha != expected:
                fail(f"frozen {lane} model SHA mismatch for seed {seed}")
            name = f"seed{seed}_{lane}"
            candidates[name] = artifact
            identities[name] = {
                "model_state_sha256": model_sha,
                "artifact_path": str(artifact),
                "artifact_model_npz_sha256": sha256_file(artifact / "model.npz"),
            }
    if (
        pr258.canonical_sha(
            {
                name: identity["model_state_sha256"]
                for name, identity in identities.items()
            }
        )
        != EXPECTED_CANDIDATE_SHA
    ):
        fail("audited candidate aggregate SHA mismatch")

    suite_paths = {
        label: workdir / "suites" / f"suite_{label}.jsonl" for label in SUITE_SEEDS
    }
    suite_audit, suite_keys, suite_prefixes = {}, {}, {}
    replay_states = set().union(
        *(set(tuple(row["state"]) for row in rows) for rows in replays.values())
    )
    for label, path in suite_paths.items():
        if not path.is_file() or sha256_file(path) != EXPECTED_SUITE_SHA[label]:
            fail(f"frozen {label} suite SHA mismatch")
        rows = pr258.suites.load_suite_jsonl(str(path))
        keys = pr258.pr249.suite_keys(rows)
        prefixes = pr258.pr251.prefix_keys(rows)
        suite_keys[label], suite_prefixes[label] = keys, prefixes
        suite_audit[label] = {
            "sha256": sha256_file(path),
            "openings": len(rows),
            "final_overlap_registry": len(
                keys
                & registry_module.final_keys(
                    {
                        key: value
                        for key, value in registry.items()
                        if key not in SUITE_SEEDS
                    }
                )
            ),
            "prefix_overlap_registry": len(
                prefixes
                & registry_module.prefix_keys(
                    {
                        key: value
                        for key, value in registry.items()
                        if key not in SUITE_SEEDS
                    }
                )
            ),
            "replay_state_overlap": sum(
                tuple(pr258.encode_state(row["state"], input_encoding="kalah_v3"))
                in replay_states
                for row in rows
            ),
        }
        if suite_audit[label]["openings"] != 128 or any(
            suite_audit[label][key]
            for key in (
                "final_overlap_registry",
                "prefix_overlap_registry",
                "replay_state_overlap",
            )
        ):
            fail(f"frozen {label} suite preflight mismatch")
    for left in SUITE_SEEDS:
        for right in SUITE_SEEDS:
            if left < right:
                final_overlap = len(suite_keys[left] & suite_keys[right])
                prefix_overlap = len(suite_prefixes[left] & suite_prefixes[right])
                if final_overlap or prefix_overlap:
                    fail(f"frozen suite mutual overlap: {left}/{right}")
                suite_audit[left][f"final_overlap_{right}"] = final_overlap
                suite_audit[left][f"prefix_overlap_{right}"] = prefix_overlap
    if not manifest["preflight"].get("passed"):
        fail("frozen preflight was not successful")

    probe_path = workdir / "search_probe.json"
    if probe_path.is_file():
        probe_results = json.loads(probe_path.read_text(encoding="utf-8"))
    else:
        probe = [row for seed in SEEDS for row in replays[seed]][:256]
        probe_results = {
            str(seed): {
                context: pr260.search_diagnostics(
                    candidates[f"seed{seed}_trunk_adapter"],
                    candidates[f"seed{seed}_policy_hidden"],
                    probe,
                    context,
                    workers,
                )
                | {"first_divergence_simulation": "not_available"}
                for context in ("384:256", "1200:1200")
            }
            for seed in SEEDS
        }
        probe_path.write_text(
            json.dumps(probe_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    contract = {
        "contexts": ["1200:1200", "384:256"],
        "c_puct": 1.25,
        "fpu_mode": "zero",
        "normalize_values": False,
        "deterministic": True,
        "seat_swap": True,
        "arena_seed": 42,
        "matched_p1_vs_p1_control_per_suite": True,
    }

    def evaluate_context(context: str) -> dict[str, Any]:
        results = {}
        for label, path in suite_paths.items():
            path = workdir / "evaluation_cache" / context / f"{label}.json"
            if path.is_file():
                results[label] = json.loads(path.read_text(encoding="utf-8"))
                continue
            value = pr260.evaluate(
                candidates, {label: suite_paths[label]}, workdir, context, workers
            )[label]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            results[label] = value
        return results

    primary_path = workdir / "primary_1200_analysis.json"
    if primary_path.is_file():
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
    else:
        primary = pr261.analyze(
            evaluate_context("1200:1200"), tuple(SUITE_SEEDS), "policy_hidden"
        )
        primary_path.write_text(
            json.dumps(primary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    shallow = pr261.analyze(
        evaluate_context("384:256"), tuple(SUITE_SEEDS), "policy_hidden"
    )
    learned = sum(
        values["heldout_beta095_ce"]["16"] < values["heldout_beta095_ce"]["initial"]
        for values in frozen["training"].values()
    )
    classification = classify(primary, learned, shallow)
    result = {
        "schema": "alphazero_lite_pr262_policy_hidden_capacity_evaluation_v1",
        "frozen_artifact_identities": identities,
        "frozen_candidate_aggregate_sha256": EXPECTED_CANDIDATE_SHA,
        "frozen_suite_audit": suite_audit,
        "frozen_manifest_hashes": {
            "suite_manifest_sha256": manifest["suite_manifest_sha256"],
            "preflight_sha256": manifest["preflight_sha256"],
        },
        "evaluation_contract": contract,
        "search_probe": probe_results,
        "primary_arena": primary,
        "shallow_arena": shallow,
        "heldout_beta095_ce_improved_seeds": learned,
        "classification": classification,
        "recommended_next_experiment": recommended_next_experiment(classification),
        "wall_clock_seconds": time.monotonic() - started,
    }
    (workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (
        REPO_ROOT / "docs/data/alphazero-lite-pr262-policy-hidden-capacity-summary.json"
    ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr262_policy_hidden_capacity")
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    args, started = parser.parse_args(), time.monotonic()
    if args.freeze_only and args.evaluate_only:
        parser.error("--freeze-only and --evaluate-only cannot be combined")
    if args.evaluate_only:
        print(evaluate_only(args.workdir, args.workers, started)["classification"])
        return
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
