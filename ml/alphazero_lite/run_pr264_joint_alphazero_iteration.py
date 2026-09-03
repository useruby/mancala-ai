#!/usr/bin/env python3
# ruff: noqa: E402
"""Prospective full-network AlphaZero iteration from frozen A16.

This runner deliberately has no selection checkpoint, no promotion path, no
shadow search, and no replay mixing.  ``--freeze-only`` is the handoff between
candidate freezing and sealed-suite evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
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
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    canonical_manifest_hash,
    configure_determinism,
    game_split,
    read_jsonl,
    sha256_file,
    write_json,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    export,
    new_model,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (
    incumbent_policy_batch,
    mixed_policy_target,
)
from ml.alphazero_lite.run_shared_trunk_delta_attribution import js
from ml.alphazero_lite.train import (
    apply_trainable_scope,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

SEEDS = (63, 64, 65, 66, 67)
LANES = {"joint_anchor95": 0.95, "joint_pure_az": 0.0}
SUITE_SEEDS = {"AK": 37042, "AL": 38042, "AM": 39042}
A16_CHECKPOINT = Path(
    "/tmp/azlite_fresh_p1_parent_adapter/artifacts/step_0016/checkpoint.npz"
)
A16_ARTIFACT = A16_CHECKPOINT.parent / "artifact"
A16_CHECKPOINT_SHA = "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34"
A16_WEIGHTS_SHA = "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789"
A16_SNAPSHOT_SHA = "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
P1_SHA = "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9"
GAMES, SIMULATIONS, BATCH_SIZE = 700, 384, 512


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def expected_registry(*, include_new_suites: bool = False) -> list[str]:
    labels = [
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
    return labels + (["AK", "AL", "AM"] if include_new_suites else [])


def verify_a16() -> dict[str, Any]:
    checks = {
        "a16_snapshot": sha256_file(replay.A16_SNAPSHOT),
        "a16_checkpoint": sha256_file(A16_CHECKPOINT),
        "a16_artifact_model": sha256_file(A16_ARTIFACT / "model.npz"),
        "a16_artifact_weights": sha256_file(A16_ARTIFACT / "weights.json"),
        "p1_checkpoint": sha256_file(replay.P1_CHECKPOINT),
    }
    expected = {
        "a16_snapshot": A16_SNAPSHOT_SHA,
        "a16_checkpoint": A16_CHECKPOINT_SHA,
        "a16_artifact_model": A16_CHECKPOINT_SHA,
        "a16_artifact_weights": A16_WEIGHTS_SHA,
        "p1_checkpoint": P1_SHA,
    }
    if checks != expected:
        fail("established A16/P1 artifact hash mismatch")
    return checks


def generate_replay(seed: int, path: Path, workers: int) -> list[dict[str, Any]]:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(REPO_ROOT / "ml/alphazero_lite/self_play.py"),
            "--out",
            str(path),
            "--games",
            str(GAMES),
            "--seed",
            str(seed),
            "--checkpoint",
            str(A16_CHECKPOINT),
            "--input-encoding",
            "kalah_v3",
            "--simulations",
            str(SIMULATIONS),
            "--c-puct",
            "1.25",
            "--player-mode",
            "puct",
            "--root-policy-mode",
            "visit_count",
            "--fpu-mode",
            "zero",
            "--tree-reuse-enabled",
            "--temperature-threshold",
            "10",
            "--temperature",
            "1.0",
            "--temperature-late",
            "0.1",
            "--dirichlet-alpha",
            "0.3",
            "--dirichlet-epsilon",
            "0.3",
            "--policy-target-mode",
            "default",
            "--policy-target-noise-mode",
            "noisy",
            "--value-target-mode",
            "default",
            "--write-game-metadata",
            "--workers",
            str(workers),
        ]
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=28800
        )
        if completed.returncode:
            raise RuntimeError(
                f"self-play seed {seed} failed: {completed.stderr[-2000:]}"
            )
    rows = read_jsonl(path)
    if set(int(row["game_index"]) for row in rows) != set(range(GAMES)):
        fail(f"seed {seed} does not contain exactly 700 games")
    # The normal self-play writer records terminal outcome under the default mode.
    if any(
        float(row["value"]) != float(row.get("final_outcome", row["value"]))
        for row in rows
    ):
        fail(f"seed {seed} value target is not terminal outcome")
    return rows


def eligible_rows(
    rows: list[dict[str, Any]], registry: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    blocked, groups = replay.exclusion_hashes(pr258.pr249.CANONICAL_SUITE)
    consumed = set().union(
        *(replay.canonical_arena_hashes(spec.path) for spec in registry.values())
    )
    return replay.filter_rows(
        rows, blocked | consumed, {**groups, "consumed_evaluation_openings": consumed}
    )


def build_manifest(
    rows: list[dict[str, Any]], seed: int, directory: Path
) -> dict[str, Any]:
    train_games, validation_games = game_split(rows, seed)
    train_set, valid_set = set(train_games), set(validation_games)
    source = np.asarray(
        [i for i, row in enumerate(rows) if int(row["game_index"]) in train_set],
        dtype=np.int64,
    )
    validation = np.asarray(
        [i for i, row in enumerate(rows) if int(row["game_index"]) in valid_set],
        dtype=np.int64,
    )
    order = np.random.default_rng(seed).permutation(len(source))
    batches = [
        order[start : start + BATCH_SIZE].tolist()
        for start in range(0, len(order), BATCH_SIZE)
    ]
    payload = {
        "schema": "azlite_pr264_joint_full_network_manifest_v1",
        "seed": seed,
        "replay_rows": len(rows),
        "train_games": train_games,
        "validation_games": validation_games,
        "train_source_indexes": source.tolist(),
        "validation_source_indexes": validation.tolist(),
        "epoch_permutation": order.tolist(),
        "batches": batches,
        "batch_size": BATCH_SIZE,
        "a16_checkpoint_sha256": A16_CHECKPOINT_SHA,
        "replay_sha256": sha256_file(directory / "replay.jsonl"),
        "recipe": {
            "epochs": 1,
            "lr": 1e-6,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "value_loss": "huber",
            "huber_delta": 1.0,
            "value_loss_weight": 1.0,
        },
    }
    payload["manifest_sha256_excluding_this_field"] = canonical_manifest_hash(payload)
    write_json(directory / "training_manifest.json", payload)
    return payload


def family(name: str) -> str:
    if name.startswith(("input_layer", "residual_layers")):
        return "input_trunk"
    if name.startswith(("policy_hidden_layer", "policy_head")):
        return "policy_hidden_readout"
    if name.startswith("policy_adapter"):
        return "adapter"
    return "value_hidden_readout"


def policy_probs(
    model: torch.nn.Module, rows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    with torch.no_grad():
        logits, values = model(torch.from_numpy(x))
    logits_np = logits.numpy().astype(np.float64)
    logits_np[~mask.astype(bool)] = -1e9
    logits_np -= logits_np.max(axis=1, keepdims=True)
    probability = np.exp(logits_np)
    return probability / probability.sum(axis=1, keepdims=True), values.numpy().reshape(
        -1
    )


def telemetry(
    model: torch.nn.Module,
    initial: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    p1: np.ndarray,
    gradient_norm: float,
    clipped: int,
    steps: int,
) -> dict[str, Any]:
    probability, values = policy_probs(model.cpu(), rows)
    a16 = new_model(torch.device("cpu"))
    a16.load_state_dict(initial)
    a16_probability, _ = policy_probs(a16, rows)
    search = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    value_target = np.asarray([row["value"] for row in rows], dtype=np.float64)
    error = values - value_target
    delta: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "total",
            "input_trunk",
            "policy_hidden_readout",
            "adapter",
            "value_hidden_readout",
        )
    }
    for name, parameter in model.state_dict().items():
        vector = (parameter.cpu() - initial[name]).reshape(-1)
        delta["total"].append(vector)
        delta[family(name)].append(vector)
    return {
        "policy_ce_own_target": float(
            np.mean(pr258._cross_entropy(probability, target))
        ),
        "policy_ce_raw_a16_mcts_target": float(
            np.mean(pr258._cross_entropy(probability, search))
        ),
        "policy_ce_p1": float(np.mean(pr258._cross_entropy(probability, p1))),
        "legal_policy_l1_vs_a16": float(
            np.abs(probability - a16_probability).sum(1).mean()
        ),
        "legal_policy_js_vs_a16": float(js(probability, a16_probability).mean()),
        "top1_disagreement_vs_a16": float(
            np.mean(np.argmax(probability, 1) != np.argmax(a16_probability, 1))
        ),
        "value_huber": float(
            np.mean(np.where(np.abs(error) < 1, 0.5 * error**2, np.abs(error) - 0.5))
        ),
        "value_mae": float(np.abs(error).mean()),
        "value_sign_accuracy": float(np.mean(np.sign(values) == np.sign(value_target))),
        "prediction_mean": float(values.mean()),
        "prediction_std": float(values.std()),
        "parameter_deltas": {
            key: float(torch.linalg.vector_norm(torch.cat(value)))
            for key, value in delta.items()
        },
        "gradient_norm": gradient_norm,
        "clipping_frequency": clipped / max(steps, 1),
    }


def gradient_audit(
    model: torch.nn.Module, batch: dict[str, torch.Tensor]
) -> dict[str, Any]:
    named = list(model.named_parameters())

    def gradients(loss: torch.Tensor) -> dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        return {
            name: parameter.grad.detach().clone()
            for name, parameter in named
            if parameter.grad is not None
        }

    logits, prediction = model(batch["x"])
    policy = compute_policy_cross_entropy(
        logits.masked_fill(batch["mask"] <= 0, -1e9), batch["p"]
    ).mean()
    value = compute_value_loss_vector(
        prediction, batch["v"], value_loss="huber", huber_delta=1.0
    ).mean()
    pg, vg = gradients(policy), gradients(value)
    result = {}
    for label in (
        "input_trunk",
        "policy_hidden_readout",
        "adapter",
        "value_hidden_readout",
    ):
        pv = torch.cat(
            [value.reshape(-1) for name, value in pg.items() if family(name) == label]
            or [torch.zeros(1)]
        )
        vv = torch.cat(
            [value.reshape(-1) for name, value in vg.items() if family(name) == label]
            or [torch.zeros(1)]
        )
        result[label] = {
            "policy_gradient_norm": float(torch.linalg.vector_norm(pv)),
            "value_gradient_norm": float(torch.linalg.vector_norm(vv)),
            "norm_ratio": float(
                torch.linalg.vector_norm(pv)
                / max(torch.linalg.vector_norm(vv).item(), 1e-12)
            ),
            "cosine": float(torch.nn.functional.cosine_similarity(pv, vv, dim=0))
            if pv.numel() == vv.numel()
            else None,
            "combined_gradient_norm": float(torch.linalg.vector_norm(pv))
            if pv.numel() != vv.numel()
            else float(torch.linalg.vector_norm(pv + vv)),
        }
    model.zero_grad(set_to_none=True)
    return result


def train_lane(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    initial: dict[str, torch.Tensor],
    p1_state: dict[str, torch.Tensor],
    beta: float,
    lane_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    configure_determinism(device, int(manifest["seed"]))
    train_rows = [rows[index] for index in manifest["train_source_indexes"]]
    validation_rows = [rows[index] for index in manifest["validation_source_indexes"]]
    p1 = new_model(device)
    p1.load_state_dict(p1_state)
    p1.eval()
    for parameter in p1.parameters():
        parameter.requires_grad_(False)
    for group in (train_rows, validation_rows):
        x = torch.from_numpy(
            np.asarray([row["state"] for row in group], dtype=np.float32)
        ).to(device)
        mask = torch.from_numpy(
            legal_mask_matrix_for_encoded_states(x.cpu().numpy())
        ).to(device)
        with torch.no_grad():
            parent = incumbent_policy_batch(p1, {"x": x, "mask": mask}).cpu().numpy()
        for row, parent_policy in zip(group, parent, strict=True):
            legal_mask = legal_mask_matrix_for_encoded_states(
                np.asarray([row["state"]], dtype=np.float32)
            )[0]
            row["target"] = (
                row["policy"]
                if beta == 0.0
                else mixed_policy_target(
                    torch.tensor([row["policy"]], device=device),
                    torch.tensor([parent_policy], device=device),
                    torch.from_numpy(legal_mask.reshape(1, -1)).to(device),
                    beta,
                )
                .cpu()
                .numpy()[0]
                .tolist()
            )
            row["p1_policy"] = parent_policy.tolist()
    model = new_model(device)
    model.load_state_dict(copy.deepcopy(initial))
    apply_trainable_scope(model, "all")
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if len(names) != len(list(model.named_parameters())):
        fail("full-network scope left parameters frozen")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.0)
    x = np.asarray([row["state"] for row in train_rows], dtype=np.float32)
    tensors = {
        "x": torch.from_numpy(x).to(device),
        "p": torch.tensor(
            np.asarray([row["target"] for row in train_rows], dtype=np.float32),
            device=device,
        ),
        "v": torch.tensor(
            np.asarray([row["value"] for row in train_rows], dtype=np.float32).reshape(
                -1, 1
            ),
            device=device,
        ),
        "mask": torch.from_numpy(legal_mask_matrix_for_encoded_states(x)).to(device),
    }
    initial_optimizer_sha = replay.optimizer_state_sha256(optimizer.state_dict())
    snapshots = {
        "initial": {
            "train": telemetry(
                model,
                initial,
                train_rows,
                np.asarray([r["p1_policy"] for r in train_rows]),
                0.0,
                0,
                0,
            ),
            "heldout": telemetry(
                model,
                initial,
                validation_rows,
                np.asarray([r["p1_policy"] for r in validation_rows]),
                0.0,
                0,
                0,
            ),
        }
    }
    audit = gradient_audit(
        model,
        {
            key: value[: min(BATCH_SIZE, len(train_rows))]
            for key, value in tensors.items()
        },
    )
    clipped = 0
    grad = 0.0
    boundaries = {
        max(
            1, round(len(manifest["batches"]) * fraction)
        ): f"epoch_{int(fraction * 100)}pct"
        for fraction in (0.25, 0.5, 0.75)
    }
    for step, indexes in enumerate(manifest["batches"], 1):
        index = torch.tensor(indexes, device=device)
        logits, prediction = model(tensors["x"][index])
        policy = compute_policy_cross_entropy(
            logits.masked_fill(tensors["mask"][index] <= 0, -1e9), tensors["p"][index]
        ).mean()
        value = compute_value_loss_vector(
            prediction, tensors["v"][index], value_loss="huber", huber_delta=1.0
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        clipped += grad > 1.0
        optimizer.step()
        if step in boundaries:
            label = boundaries[step]
            snapshots[label] = {
                "train": telemetry(
                    model,
                    initial,
                    train_rows,
                    np.asarray([r["p1_policy"] for r in train_rows]),
                    grad,
                    clipped,
                    step,
                ),
                "heldout": telemetry(
                    model,
                    initial,
                    validation_rows,
                    np.asarray([r["p1_policy"] for r in validation_rows]),
                    grad,
                    clipped,
                    step,
                ),
            }
    snapshots["final_epoch1"] = {
        "train": telemetry(
            model,
            initial,
            train_rows,
            np.asarray([r["p1_policy"] for r in train_rows]),
            grad,
            clipped,
            len(manifest["batches"]),
        ),
        "heldout": telemetry(
            model,
            initial,
            validation_rows,
            np.asarray([r["p1_policy"] for r in validation_rows]),
            grad,
            clipped,
            len(manifest["batches"]),
        ),
    }
    state = state_copy(model.cpu())
    artifact = export(state, lane_dir, f"pr264_seed{manifest['seed']}_{lane_dir.name}")
    return {
        "artifact": str(artifact),
        "artifact_model_sha256": sha256_file(artifact / "model.npz"),
        "model_state_sha256": pr258.contract.state_sha256(state),
        "optimizer_initial_sha256": initial_optimizer_sha,
        "trainable_parameter_names": names,
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "gradient_audit": audit,
        "telemetry": snapshots,
        "value_target_sha256": pr258.canonical_sha(
            [row["value"] for row in train_rows + validation_rows]
        ),
    }


def direct_evaluate(
    candidates: dict[str, Path],
    suites: dict[str, Path],
    workdir: Path,
    context: str,
    workers: int,
) -> dict[str, Any]:
    from ml.alphazero_lite.run_pr241_policy_target_noise_isolation import arena_records

    result = {}
    for label, suite in suites.items():
        control = arena_records(
            workdir / "arena" / context / label,
            A16_ARTIFACT,
            A16_ARTIFACT,
            context,
            "a16_control",
            workers,
            suite,
        )
        result[label] = {}
        for name, candidate in candidates.items():
            records = arena_records(
                workdir / "arena" / context / label,
                candidate,
                A16_ARTIFACT,
                context,
                name,
                workers,
                suite,
            )
            result[label][name] = {
                "effect": pr258.paired_opening_candidate_effect(records, control)
            }
    return result


def analyze(raw: dict[str, Any], lane: str) -> dict[str, Any]:
    per_seed, nested = {}, []
    for seed in SEEDS:
        suite_vectors, suite_effects = [], {}
        for label in SUITE_SEEDS:
            effect = raw[label][f"seed{seed}_{lane}"]["effect"]
            vector = np.asarray(
                [
                    effect["per_opening_effect"][key]
                    for key in sorted(effect["per_opening_effect"])
                ]
            )
            suite_vectors.append(vector)
            suite_effects[label] = float(vector.mean())
        vector = np.concatenate(suite_vectors)
        per_seed[str(seed)] = {
            "suite_effects": suite_effects,
            "effect": float(vector.mean()),
        }
        nested.append(suite_vectors)
    values = np.asarray([per_seed[str(seed)]["effect"] for seed in SEEDS])
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(10000):
        draws.append(
            float(
                np.mean(
                    [
                        rng.choice(
                            nested[i][rng.integers(0, 3)], 128, replace=True
                        ).mean()
                        for i in rng.integers(0, 5, 5)
                    ]
                )
            )
        )
    return {
        "per_seed": per_seed,
        "pooled": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "sd": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_seed_count": int((values > 0).sum()),
            "hierarchical_ci95": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        },
    }


def classification(primary: dict[str, Any], shallow: dict[str, Any]) -> str:
    def passed(result: dict[str, Any]) -> bool:
        values = [item["effect"] for item in result["per_seed"].values()]
        p = result["pooled"]
        return (
            p["mean"] > 0
            and p["hierarchical_ci95"][0] > 0
            and p["positive_seed_count"] >= 4
            and min(values) >= -0.02
        )

    pure, anchor = passed(primary["joint_pure_az"]), passed(primary["joint_anchor95"])
    shallow_harm = any(
        result["pooled"]["mean"] < -0.02
        or sum(item["effect"] < -0.02 for item in result["per_seed"].values()) >= 2
        for result in shallow.values()
    )
    if pure and not shallow_harm and not anchor:
        return "pure_joint_alphazero_iteration_wins"
    if anchor and not pure:
        return "joint_training_works_but_parent_anchor_required"
    if pure and anchor:
        return (
            "joint_update_high_budget_only"
            if shallow_harm
            else "joint_training_helps_independent_of_anchor"
        )
    if primary["joint_pure_az"]["pooled"]["mean"] > 0:
        return "pure_target_helps_but_is_seed_sensitive"
    if all(result["pooled"]["mean"] < -0.02 for result in primary.values()):
        return "joint_full_network_degrades"
    return "joint_full_network_does_not_beat_incumbent"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr264_joint_alphazero_iteration"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    args.workdir.mkdir(parents=True, exist_ok=True)
    registry = registry_module.load(args.workdir)
    if list(registry) != expected_registry(include_new_suites=args.evaluate_only):
        fail("consumed-suite registry is not authoritative through AJ")
    hashes = verify_a16()
    if not args.evaluate_only:
        snapshot = torch.load(
            replay.A16_SNAPSHOT, map_location="cpu", weights_only=False
        )
        initial, _ = replay.immutable_initial_state(snapshot)
        p1_model = new_model(torch.device("cpu"))
        load_checkpoint_into_model(p1_model, replay.P1_CHECKPOINT)
        p1_state = state_copy(p1_model)
        frozen, all_replays = {"a16_hashes": hashes, "seeds": {}}, {}
        for seed in SEEDS:
            directory = args.workdir / f"seed{seed}"
            generated_path = directory / "generated" / "ordinary_reused.jsonl"
            replay_path = directory / "replay.jsonl"
            raw = generate_replay(seed, generated_path, args.workers)
            rows, exclusions = eligible_rows(raw, registry)
            if not rows:
                fail(f"seed {seed} has no eligible rows")
            # Training must use the filtered persisted replay, never a mixed source.
            directory.mkdir(parents=True, exist_ok=True)
            replay_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest = build_manifest(rows, seed, directory)
            all_replays[seed] = rows
            lanes = {
                name: train_lane(
                    copy.deepcopy(rows),
                    manifest,
                    initial,
                    p1_state,
                    beta,
                    directory / name,
                    torch.device("cpu"),
                )
                for name, beta in LANES.items()
            }
            if (
                lanes["joint_anchor95"]["value_target_sha256"]
                != lanes["joint_pure_az"]["value_target_sha256"]
            ):
                fail(f"seed {seed} lanes have different value targets")
            if any(
                info["trainable_parameter_count"] != info["total_parameter_count"]
                for info in lanes.values()
            ):
                fail(f"seed {seed} scope is not all")
            frozen["seeds"][str(seed)] = {
                "raw_replay_sha256": sha256_file(generated_path),
                "replay_sha256": sha256_file(replay_path),
                "exclusions": exclusions,
                "manifest_sha256": manifest["manifest_sha256_excluding_this_field"],
                "lanes": lanes,
            }
        frozen["candidate_aggregate_sha256"] = pr258.canonical_sha(
            {
                f"seed{seed}_{lane}": frozen["seeds"][str(seed)]["lanes"][lane][
                    "model_state_sha256"
                ]
                for seed in SEEDS
                for lane in LANES
            }
        )
        write_json(args.workdir / "frozen_candidates.json", frozen)
        pr258.SUITE_SEEDS = SUITE_SEEDS
        suites, suite_manifest, preflight = pr258.seal_suites(
            args.workdir, registry, all_replays
        )
        if not preflight["passed"]:
            fail("AK/AL/AM preflight failed")
        write_json(
            args.workdir / "frozen_manifest.json",
            frozen
            | {
                "suite_manifest": suite_manifest,
                "preflight": preflight,
                "suite_manifest_sha256": pr258.canonical_sha(suite_manifest),
                "preflight_sha256": pr258.canonical_sha(preflight),
            },
        )
        if args.freeze_only:
            return
    frozen = json.loads(
        (args.workdir / "frozen_candidates.json").read_text(encoding="utf-8")
    )
    suites = {
        label: args.workdir / "suites" / f"suite_{label}.jsonl" for label in SUITE_SEEDS
    }
    candidates = {
        f"seed{seed}_{lane}": Path(
            frozen["seeds"][str(seed)]["lanes"][lane]["artifact"]
        )
        for seed in SEEDS
        for lane in LANES
    }
    primary_raw = direct_evaluate(
        candidates, suites, args.workdir, "1200:1200", args.workers
    )
    primary = {lane: analyze(primary_raw, lane) for lane in LANES}
    shallow_raw = direct_evaluate(
        candidates, suites, args.workdir, "384:384", args.workers
    )
    shallow = {lane: analyze(shallow_raw, lane) for lane in LANES}
    result = frozen | {
        "primary_1200": primary,
        "shallow_384": shallow,
        "classification": classification(primary, shallow),
        "wall_clock_seconds": time.monotonic() - started,
    }
    write_json(args.workdir / "summary.json", result)
    print(result["classification"])


if __name__ == "__main__":
    main()
