#!/usr/bin/env python3
# ruff: noqa: E402
"""Train the fixed A16 continuation on matched ordinary and shadow self-play."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import canonical_game_state_hash
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    game_split,
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_adapter_margin_sensitivity import (
    HELD_OUT_HASHES,
)
from ml.alphazero_lite.run_fresh_p1_adapter_matched_q_feedback import (
    FROZEN,
    PR222,
    _control_subset,
    decode_kalah_v3_base_state,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    ADAPTER_KEYS,
    export,
    new_model,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (
    _cross_entropy,
    incumbent_policy_batch,
    mixed_policy_target,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch, _losses
from ml.alphazero_lite.run_shared_trunk_delta_attribution import js
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.train import (
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

SEED = 44
BETA = 0.95
STEPS = (1, 4, 16)
A16_SNAPSHOT = Path(
    "/tmp/azlite_fresh_p1_parent_adapter/beta095/snapshots/step_0016.pt"
)
P1_CHECKPOINT = Path(
    "/tmp/azlite_fresh_selfplay_anchor/beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
)
A16_WEIGHTS_SHA = "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789"
P1_CHECKPOINT_SHA = "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9"


def optimizer_state_sha256(state: dict[str, Any]) -> str:
    """Hash an optimizer state without depending on Python object identity."""
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict\0")
            for key in sorted(
                value, key=lambda item: (type(item).__name__, repr(item))
            ):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"sequence\0")
            for item in value:
                update(item)
        elif value is None:
            digest.update(b"none\0")
        else:
            digest.update(
                f"{type(value).__name__}:{json.dumps(value, sort_keys=True)}\0".encode(
                    "ascii"
                )
            )

    update(state)
    return digest.hexdigest()


def immutable_initial_state(
    snapshot: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Detach the A16 snapshot from ``torch.load`` storage before any lane starts."""
    return (
        {
            name: value.detach().cpu().clone()
            for name, value in snapshot["model"].items()
        },
        copy.deepcopy(snapshot["optimizer"]),
    )


def load_isolated_optimizer(
    model: torch.nn.Module, optimizer_state: dict[str, Any]
) -> torch.optim.Adam:
    """Create Adam from a private copy; Adam otherwise retains state tensor references."""
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-5,
    )
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    return optimizer


def canonical_arena_hashes(suite_path: Path) -> set[str]:
    hashes = set()
    for entry in read_jsonl(suite_path):
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        for move in entry["prefix_moves"]:
            if not game.move(int(move)):
                raise RuntimeError("canonical suite contains an illegal opening move")
        hashes.add(canonical_game_state_hash(game))
    if len(hashes) != 128:
        raise RuntimeError("canonical suite must contain 128 distinct opening states")
    return hashes


def exclusion_hashes(suite_path: Path) -> tuple[set[str], dict[str, set[str]]]:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    prior = json.loads(PR222.read_text(encoding="utf-8"))
    amplified = set(frozen["full_amplified_1200"])
    prior_by_hash = {row["state_hash"]: row for row in prior["records"]}
    washed = {
        row["state_hash"]
        for row in _control_subset(
            prior["records"], [prior_by_hash[key] for key in amplified]
        )
    }
    groups = {
        "frozen_amplified": amplified,
        "frozen_washed": washed,
        "held_out": set(HELD_OUT_HASHES),
        "canonical_arena": canonical_arena_hashes(suite_path),
    }
    return set().union(*groups.values()), groups


def filter_rows(
    rows: list[dict[str, Any]], blocked: set[str], groups: dict[str, set[str]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = Counter({name: 0 for name in groups})
    kept = []
    for row in rows:
        state = decode_kalah_v3_base_state(list(row["state"]))
        digest = canonical_game_state_hash(KalahGame.from_state(state))
        reason = next(
            (name for name, values in groups.items() if digest in values), None
        )
        if reason is None:
            kept.append(row)
        else:
            counts[reason] += 1
    counts["total_excluded"] = len(rows) - len(kept)
    counts["eligible_rows"] = len(kept)
    return kept, dict(counts)


def policy(state: dict[str, torch.Tensor], rows: list[dict[str, Any]]) -> np.ndarray:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    x = torch.tensor(np.asarray([row["state"] for row in rows], dtype=np.float32))
    mask = legal_mask_matrix_for_encoded_states(x.numpy())
    with torch.no_grad():
        logits, _ = model(x)
    values = logits.numpy().astype(np.float64)
    values[~mask.astype(bool)] = -1e9
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def batches(rows: list[dict[str, Any]]) -> list[np.ndarray]:
    train_games, _validation_games = game_split(rows, SEED)
    source = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["game_index"] in set(train_games)
        ],
        dtype=np.int64,
    )
    plan = np.random.default_rng(SEED).permutation(len(source))
    return [source[plan[index : index + 512]] for index in range(0, len(plan), 512)]


def train_lane(
    rows: list[dict[str, Any]],
    initial: dict[str, Any],
    parent_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]]:
    model = new_model(device)
    model.load_state_dict(initial["model"])
    apply_trainable_scope(model, "policy_adapter_only")
    optimizer_before = optimizer_state_sha256(initial["optimizer"])
    optimizer = load_isolated_optimizer(model, initial["optimizer"])
    parent = new_model(device)
    parent.load_state_dict(parent_state)
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    snapshots, optimizer_states = {}, {}
    model.train()
    for step, indexes in enumerate(batches(rows)[:16], 1):
        batch = _batch(rows, indexes, device)
        target = mixed_policy_target(
            batch["p"], incumbent_policy_batch(parent, batch), batch["mask"], BETA
        )
        policy_loss, value_loss = _losses(model, {**batch, "p": target})
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        if step in STEPS:
            snapshots[step] = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            optimizer_states[step] = copy.deepcopy(optimizer.state_dict())
    if optimizer_state_sha256(initial["optimizer"]) != optimizer_before:
        raise RuntimeError("invariant_failure: train_lane mutated its input optimizer")
    return snapshots, optimizer_states


def metrics(
    rows: list[dict[str, Any]],
    snapshots: dict[int, dict[str, torch.Tensor]],
    parent: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
    initial_optimizer: dict[str, Any],
) -> dict[str, Any]:
    search = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    parent_policy = policy(parent, rows)
    optimizer_before = optimizer_state_sha256(initial_optimizer)
    pure, _unused = train_lane(
        rows,
        {
            "model": copy.deepcopy(initial),
            "optimizer": copy.deepcopy(initial_optimizer),
        },
        parent,
        torch.device("cpu"),
    )
    if optimizer_state_sha256(initial_optimizer) != optimizer_before:
        raise RuntimeError("invariant_failure: metrics mutated its input optimizer")
    pure_ce = float(np.mean(_cross_entropy(policy(pure[16], rows), search)))
    parent_ce = float(np.mean(_cross_entropy(parent_policy, search)))
    result = {}
    for step, state in snapshots.items():
        candidate = policy(state, rows)
        ce = float(np.mean(_cross_entropy(candidate, search)))
        adapter_delta = torch.cat(
            [(state[key] - initial[key]).reshape(-1) for key in ADAPTER_KEYS]
        )
        result[str(step)] = {
            "ce_search": ce,
            "ce_p1": float(np.mean(_cross_entropy(candidate, parent_policy))),
            "ce_beta095": float(
                np.mean(_cross_entropy(candidate, 0.05 * search + 0.95 * parent_policy))
            ),
            "fit_fraction": (parent_ce - ce) / (parent_ce - pure_ce),
            "legal_policy_l1_vs_p1": float(
                np.abs(candidate - parent_policy).sum(axis=1).mean()
            ),
            "js_vs_p1": float(js(candidate, parent_policy).mean()),
            "top1_disagreement": float(
                np.mean(
                    np.argmax(candidate, axis=1) != np.argmax(parent_policy, axis=1)
                )
            ),
            "adapter_norm": float(
                torch.linalg.vector_norm(
                    torch.cat([state[key].reshape(-1) for key in ADAPTER_KEYS])
                )
            ),
            "delta_from_a16": float(torch.linalg.vector_norm(adapter_delta)),
            "inherited_parameters_bit_identical": all(
                torch.equal(state[key], initial[key])
                for key in state
                if key not in ADAPTER_KEYS
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_onpolicy_shadow_replay")
    )
    parser.add_argument(
        "--canonical-suite",
        type=Path,
        default=Path("/tmp/azlite_opening_suite/medium_eval.jsonl"),
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if sha256_file(P1_CHECKPOINT) != P1_CHECKPOINT_SHA:
        raise RuntimeError("P1 checkpoint hash mismatch")
    initial = torch.load(A16_SNAPSHOT, map_location="cpu", weights_only=False)
    p1 = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1, P1_CHECKPOINT)
    parent = {
        name: value.detach().cpu().clone() for name, value in p1.state_dict().items()
    }
    blocked, groups = exclusion_hashes(args.canonical_suite)
    result: dict[str, Any] = {
        "schema": "azlite_onpolicy_shadow_replay_v1",
        "beta": BETA,
        "lr": 1e-5,
        "steps": list(STEPS),
        "replay_buffer": "newly_generated_replay_only",
        "canonical_suite_sha256": sha256_file(args.canonical_suite),
        "canonical_suite_replacement": True,
        "lanes": {},
    }
    for lane in ("ordinary_onpolicy", "shadow075_onpolicy", "shadow100_onpolicy"):
        source = args.workdir / f"{lane}.jsonl"
        rows, exclusions = filter_rows(read_jsonl(source), blocked, groups)
        snapshots, optimizer_states = train_lane(
            rows, initial, parent, torch.device("cpu")
        )
        lane_dir = args.workdir / f"{lane}_train"
        lane_dir.mkdir(exist_ok=True)
        for step, state in snapshots.items():
            torch.save(
                {"model": state, "optimizer": optimizer_states[step]},
                lane_dir / f"step_{step:04d}.pt",
            )
            export(state, lane_dir / "artifacts" / f"step_{step:04d}", f"{lane}_{step}")
        result["lanes"][lane] = {
            "replay_sha256": sha256_file(source),
            "exclusions": exclusions,
            "metrics": metrics(
                rows, snapshots, parent, initial["model"], initial["optimizer"]
            ),
        }
    if args.evaluate:
        from ml.alphazero_lite import (
            run_fresh_p1_adapter_lagged_parent_shadow_q as arena,
        )

        arena.ARENA_SUITE = args.canonical_suite
        p1_artifact = P1_CHECKPOINT.parent / "artifact"
        controls = {
            context: arena.arena_records(
                args.workdir / "arena",
                p1_artifact,
                p1_artifact,
                None,
                context,
                "p1_control",
                args.workers,
            )
            for context in ("384:256", "1200:1200")
        }
        result["ordinary_puct_arena"] = {}
        for lane in result["lanes"]:
            lane_results = {}
            for step in STEPS:
                candidate = (
                    args.workdir / f"{lane}_train/artifacts/step_{step:04d}/artifact"
                )
                lane_results[str(step)] = {}
                for context, control in controls.items():
                    records = arena.arena_records(
                        args.workdir / "arena",
                        candidate,
                        p1_artifact,
                        None,
                        context,
                        f"{lane}_{step}",
                        args.workers,
                    )
                    effect = paired_opening_candidate_effect(records, control)
                    ci = effect["opening_bootstrap_ci"]
                    lane_results[str(step)][context] = {
                        "effect": effect["paired_candidate_effect"],
                        "ci": ci,
                        "safe": ci["upper_95"] >= 0.0 or ci["lower_95"] >= -0.03,
                    }
            result["ordinary_puct_arena"][lane] = lane_results
    args.workdir.joinpath("training_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
