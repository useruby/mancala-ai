#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit PR #191's first joint update across deterministic whole-game shards.

This diagnostic never changes replay rows, targets, optimization settings, or
search settings. Each virtual Adam step starts from the fresh PR #191 state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.evaluation_seed_contract import (
    SEED_CONTRACT_VERSION,
    stable_seed,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    HIDDEN_SIZES,
    MODEL_TYPE,
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
    write_fixed_npz,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (
    run_arena,
    parse_game_jsonl,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch, _losses
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (
    _context_c_puct,
    _search_seed,
    _visit_policy,
    decoded_validation_manifest,
    js,
    model_outputs,
    output_metrics,
    stable_hash,
)
from ml.alphazero_lite.run_terminal_outcome_selfplay_iteration_smoke import (
    export_checkpoint,
)
from ml.alphazero_lite.self_play import build_eval_search_options
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    input_size_for_encoding,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

SHARDS = ("S0", "S1", "S2", "S3")
SIGNALS = ("policy", "value", "joint")
GROUPS = ("shared_trunk", "policy_private_head", "value_private_head")
CONTEXTS = ("384:256", "768:768", "1200:1200")
CURRENT_HASH = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
ARENA_SUITE = Path(
    "/tmp/azlite_shared_trunk_learning/arena_vs_current/temp_0_0/seed_42/artifact/equal_high/starts_0/opening_suite.jsonl"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def new_model(device: torch.device) -> PolicyValueNet:
    return PolicyValueNet(
        HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding("kalah_v3")
    ).to(device)


def parameter_group(name: str) -> str:
    if name.startswith(("input_layer.", "residual_layers.")):
        return "shared_trunk"
    if name.startswith(("policy_hidden_layer.", "policy_head.")):
        return "policy_private_head"
    if name.startswith(("value_hidden_layer.", "value_head.")):
        return "value_private_head"
    raise ValueError(f"unclassified parameter: {name}")


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.dot(left, right)
        / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right) + 1e-20)
    )


def game_identity(row: dict[str, Any]) -> str:
    """Use source game index plus immutable trajectory identity, not state order."""
    return f"{row['game_index']}:{row.get('trajectory_hash', '')}"


def phase(row: dict[str, Any]) -> str:
    remaining = sum(float(value) for value in row["state"][:12])
    return "late" if remaining <= 0.25 else "mid" if remaining <= 0.5 else "opening"


def partition(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    games: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        games[game_identity(row)].append(index)
    # Greedy longest-first bin packing balances rows while retaining whole games.
    assigned = {shard: [] for shard in SHARDS}
    loads = {shard: 0 for shard in SHARDS}
    for identity, indexes in sorted(
        games.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        shard = min(SHARDS, key=lambda name: (loads[name], name))
        assigned[shard].extend(indexes)
        loads[shard] += len(indexes)
    state_sets = {
        name: {stable_hash(rows[index]["state"]) for index in indexes}
        for name, indexes in assigned.items()
    }
    manifest: dict[str, Any] = {
        "partition_method": "greedy_whole_game_row_balance",
        "shards": {},
    }
    for name, indexes in assigned.items():
        shard_rows = [rows[index] for index in indexes]
        identities = sorted({game_identity(row) for row in shard_rows})
        entropies = [
            -float(
                np.sum(
                    np.asarray(row["policy"])
                    * np.log(np.clip(row["policy"], 1e-12, None))
                )
            )
            for row in shard_rows
        ]
        manifest["shards"][name] = {
            "source_row_indexes": sorted(indexes),
            "game_identities": identities,
            "row_count": len(indexes),
            "game_count": len(identities),
            "player_distribution": dict(
                Counter(str(row["player"]) for row in shard_rows)
            ),
            "phase_distribution": dict(Counter(phase(row) for row in shard_rows)),
            "value_target_distribution": dict(
                Counter(str(row["value"]) for row in shard_rows)
            ),
            "policy_target_entropy": {
                "mean": float(np.mean(entropies)),
                "median": float(np.median(entropies)),
            },
            "shard_sha256": sha256_bytes(
                json.dumps(
                    {"games": identities, "rows": sorted(indexes)}, sort_keys=True
                ).encode()
            ),
        }
    manifest["state_overlap"] = {
        f"{left}<->{right}": len(state_sets[left] & state_sets[right])
        for offset, left in enumerate(SHARDS)
        for right in SHARDS[offset + 1 :]
    }
    manifest["game_overlap"] = {
        f"{left}<->{right}": 0
        for offset, left in enumerate(SHARDS)
        for right in SHARDS[offset + 1 :]
    }
    return assigned, manifest


def deterministic_batches(indexes: list[int], shard: str) -> list[np.ndarray]:
    rng = np.random.default_rng(stable_seed("azlite_game_shard_batches_v1", shard, 191))
    return [
        rng.choice(np.asarray(indexes), size=512, replace=len(indexes) < 512)
        for _ in range(32)
    ]


def vectors(
    model: PolicyValueNet, batch: dict[str, torch.Tensor]
) -> dict[str, dict[str, torch.Tensor]]:
    policy_loss, value_loss = _losses(model, batch)
    named = list(model.named_parameters())
    parameters = [value for _, value in named]
    policy = torch.autograd.grad(
        policy_loss, parameters, retain_graph=True, allow_unused=True
    )
    value = torch.autograd.grad(value_loss, parameters, allow_unused=True)
    result = {group: {} for group in GROUPS}
    for group in GROUPS:
        selected = [
            (parameter, p, v)
            for (name, parameter), p, v in zip(named, policy, value, strict=True)
            if parameter_group(name) == group
        ]
        p = torch.cat(
            [
                (gradient if gradient is not None else torch.zeros_like(parameter))
                .detach()
                .reshape(-1)
                for parameter, gradient, _ in selected
            ]
        )
        v = torch.cat(
            [
                (gradient if gradient is not None else torch.zeros_like(parameter))
                .detach()
                .reshape(-1)
                for parameter, _, gradient in selected
            ]
        )
        result[group] = {"policy": p, "value": v, "joint": p + v}
    return result


def virtual_vectors(
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    lr: float,
    clip: float,
) -> dict[str, dict[str, torch.Tensor]]:
    result = {group: {} for group in GROUPS}
    for signal in SIGNALS:
        model = new_model(device)
        model.load_state_dict(state)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        before = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        policy_loss, value_loss = _losses(model, batch)
        optimizer.zero_grad(set_to_none=True)
        {"policy": policy_loss, "value": value_loss, "joint": policy_loss + value_loss}[
            signal
        ].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        for group in GROUPS:
            result[group][signal] = torch.cat(
                [
                    (value.detach() - before[name]).reshape(-1)
                    for name, value in model.named_parameters()
                    if parameter_group(name) == group
                ]
            )
    return result


def pairwise(
    samples: dict[str, list[dict[str, dict[str, torch.Tensor]]]], *, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for group in GROUPS:
        result[group] = {}
        for signal in SIGNALS:
            values = {}
            for left_i, left in enumerate(SHARDS):
                for right in SHARDS[left_i + 1 :]:
                    raw = np.asarray(
                        [
                            cosine(a[group][signal], b[group][signal])
                            for a, b in zip(samples[left], samples[right], strict=True)
                        ]
                    )
                    draws = raw[
                        rng.integers(0, len(raw), size=(10_000, len(raw)))
                    ].mean(1)
                    values[f"{left}<->{right}"] = {
                        "mean": float(raw.mean()),
                        "median": float(np.median(raw)),
                        "lower_95": float(np.quantile(draws, 0.025)),
                        "upper_95": float(np.quantile(draws, 0.975)),
                    }
            result[group][signal] = values
    return result


def conflict_stability(
    samples: dict[str, list[dict[str, dict[str, torch.Tensor]]]], *, seed: int
) -> dict[str, Any]:
    """Within-shard policy/value conflict, bootstrapped across batches per shard."""
    rng = np.random.default_rng(seed)
    per_shard: dict[str, Any] = {}
    for shard in SHARDS:
        per_shard[shard] = {}
        for group in GROUPS:
            values = np.asarray(
                [
                    cosine(sample[group]["policy"], sample[group]["value"])
                    for sample in samples[shard]
                ]
            )
            draws = values[
                rng.integers(0, len(values), size=(10_000, len(values)))
            ].mean(1)
            per_shard[shard][group] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "lower_95": float(np.quantile(draws, 0.025)),
                "upper_95": float(np.quantile(draws, 0.975)),
            }
    stability: dict[str, Any] = {}
    for group in GROUPS:
        medians = [per_shard[shard][group]["median"] for shard in SHARDS]
        stability[group] = {
            "median_range": float(max(medians) - min(medians)),
            "all_same_sign": all(m > 0 for m in medians) or all(m < 0 for m in medians),
            "conflict_sign": (
                "positive"
                if all(m > 0 for m in medians)
                else "negative"
                if all(m < 0 for m in medians)
                else "mixed"
            ),
            "medians": {shard: medians[i] for i, shard in enumerate(SHARDS)},
        }
    return {"per_shard": per_shard, "stability": stability}


def pairwise_output_agreement(
    probe_rows: list[dict[str, Any]],
    candidate_states: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Measure pairwise output agreement between the four step-1 candidates."""
    x = np.asarray([row["state"] for row in probe_rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x).astype(bool)
    outputs = {
        name: model_outputs(state, x, mask) for name, state in candidate_states.items()
    }
    names = list(candidate_states)
    result: dict[str, Any] = {}
    for left_i, left in enumerate(names):
        for right in names[left_i + 1 :]:
            left_policy, left_value = outputs[left]
            right_policy, right_value = outputs[right]
            kl = np.sum(
                left_policy
                * np.log(
                    np.clip(left_policy, 1e-12, None)
                    / np.clip(right_policy, 1e-12, None)
                ),
                axis=1,
            )
            result[f"{left}<->{right}"] = {
                "policy_kl_mean": float(np.mean(kl)),
                "policy_top1_agreement": float(
                    np.mean(
                        np.argmax(left_policy, axis=1)
                        == np.argmax(right_policy, axis=1)
                    )
                ),
                "value_mae_delta": float(np.mean(np.abs(left_value - right_value))),
                "value_sign_agreement": float(
                    np.mean(np.sign(left_value) == np.sign(right_value))
                ),
            }
    return result


def state_direction_agreement(
    cached: dict[tuple[str, str, str], dict[str, Any]],
    context: str,
    candidates: dict[str, dict[str, torch.Tensor]],
    probe: list[dict[str, Any]],
) -> dict[str, Any]:
    """Agreement metrics using functional move semantics, not value-sign as "direction"."""
    names = list(candidates)
    move = {name: [] for name in names}
    root_delta_sign = {name: [] for name in names}
    for row in probe:
        base = cached[("C", context, row["state_hash"])]
        for name in names:
            record = cached[(name, context, row["state_hash"])]
            move[name].append(int(record["move"]))
            root_delta_sign[name].append(
                int(np.sign(record["root_value"] - base["root_value"]))
            )
    move_arr = np.asarray([move[name] for name in names], dtype=np.int64)
    root_delta_arr = np.asarray(
        [root_delta_sign[name] for name in names], dtype=np.int64
    )
    base_moves = np.asarray(
        [int(cached[("C", context, row["state_hash"])]["move"]) for row in probe],
        dtype=np.int64,
    )
    change_arr = np.asarray(move_arr != base_moves[None, :], dtype=np.int64)
    # Same state change/no-change agreement across every candidate pair.
    pairwise_change_rates = [
        float(np.mean(change_arr[i] == change_arr[j]))
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    # Absolute selected-move agreement across every candidate pair.
    pairwise_move_rates = [
        float(np.mean(move_arr[i] == move_arr[j]))
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    # States where >=2 candidates change their selected move; among those,
    # whether every changing candidate selects the same NEW move, and whether
    # their root-value delta signs agree (reported separately, never as "move direction").
    any_changed = change_arr.sum(axis=0) >= 2
    same_new_move = []
    same_root_sign = []
    for state_index in np.where(any_changed)[0]:
        changed_mask = change_arr[:, state_index] == 1
        new_moves = move_arr[changed_mask, state_index]
        same_new_move.append(int(np.all(new_moves == new_moves[0])))
        root_signs = root_delta_arr[changed_mask, state_index]
        same_root_sign.append(int(np.all(root_signs == root_signs[0])))
    return {
        "same_changed_state_rate": float(
            np.mean(np.all(change_arr == change_arr[0], axis=0))
        ),
        "pairwise_change_agreement_rate_mean": float(np.mean(pairwise_change_rates)),
        "pairwise_selected_move_agreement": float(np.mean(pairwise_move_rates)),
        "same_new_move_rate": (float(np.mean(same_new_move)) if same_new_move else 0.0),
        "root_value_delta_sign_agreement": (
            float(np.mean(same_root_sign)) if same_root_sign else 0.0
        ),
        "states_changed_by_any_candidate": int(np.sum(change_arr.sum(axis=0) >= 1)),
        "states_changed_by_all_candidates": int(
            np.sum(np.all(change_arr == 1, axis=0))
        ),
    }


def fresh_state(
    manifest: dict[str, Any], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model = new_model(device)
    load_checkpoint_into_model(
        model, Path(manifest["artifact_paths"]["initialization_checkpoint"])
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(manifest["optimizer"]["lr"]),
        weight_decay=float(manifest["optimizer"].get("weight_decay", 0.0)),
    )
    return (
        {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        copy.deepcopy(optimizer.state_dict()),
    )


def real_candidate(
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    manifest: dict[str, Any],
) -> dict[str, torch.Tensor]:
    model = new_model(device)
    model.load_state_dict(state)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(manifest["optimizer"]["lr"])
    )
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    policy, value = _losses(model, batch)
    optimizer.zero_grad(set_to_none=True)
    (policy + value).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(manifest["gradient_clip"]))
    optimizer.step()
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def export_artifact(state: dict[str, torch.Tensor], name: str, workdir: Path) -> Path:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(state)
    directory = workdir / "candidates" / name
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.npz"
    write_fixed_npz(checkpoint, checkpoint_from_model(model))
    artifact = directory / "artifact"
    export_checkpoint(
        checkpoint_path=checkpoint,
        out_dir=artifact,
        version=name,
        policy_loss=0.0,
        value_loss=0.0,
    )
    metadata = json.loads((artifact / "metadata.json").read_text())
    metadata.update({"diagnostic_only": True, "promotion_forbidden": True})
    (artifact / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return artifact


def puct_record(
    row: dict[str, Any],
    evaluator: arena.ArtifactEvaluator,
    context: str,
    manifest_hash: str,
) -> dict[str, Any]:
    seed, _ = _search_seed(row, manifest_hash, 191)
    value = arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=row["state"],
        simulations=int(context.split(":")[0]),
        seed=seed,
        c_puct=_context_c_puct(context),
        search_options=build_eval_search_options(
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
            normalize_values=False,
        ),
    )
    stats = list(value["child_stats"])
    visits = sorted((int(item["visits"]) for item in stats), reverse=True)
    return {
        "move": int(value["selected_move"]),
        "visit": _visit_policy(stats),
        "q_ranking_move_ids": [
            int(item["move"])
            for item in sorted(
                stats, key=lambda item: (-float(item["q_value"]), int(item["move"]))
            )
        ],
        "root_value": float(value.get("search_root_value", value["value"])),
        "margin": visits[0] - visits[1] if len(visits) > 1 else visits[0],
    }


def classify(summary: dict[str, Any]) -> dict[str, str]:
    joint = [
        item["median"]
        for item in summary["phase_c"]["pairwise_cosines"]["shared_trunk"][
            "joint"
        ].values()
    ]
    value = [
        item["median"]
        for item in summary["phase_b"]["pairwise_cosines"]["shared_trunk"][
            "value"
        ].values()
    ]
    policy = [
        item["median"]
        for item in summary["phase_b"]["pairwise_cosines"]["shared_trunk"][
            "policy"
        ].values()
    ]
    arena = [
        item["paired_candidate_effect"]
        for item in summary.get("phase_g", {})
        .get("arena", {})
        .get("384:256", {})
        .values()
    ]
    arena_complete = len(arena) == len(SHARDS)
    similar = arena_complete and max(arena) - min(arena) < 0.05
    if (
        np.median(joint) >= 0.75
        and len(arena) == 4
        and all(item < 0 for item in arena)
        and sum(
            item["opening_bootstrap_ci"]["upper_95"] < 0
            for item in summary["phase_g"]["arena"]["384:256"].values()
        )
        >= 3
        and similar
    ):
        return {
            "label": "harmful_supervised_direction_replicated",
            "next_action": "audit supervised objective and targets",
        }
    if np.median(value) < 0.50 and np.median(policy) >= 0.50:
        return {
            "label": "value_gradient_sampling_instability",
            "next_action": "investigate outcome-target variance/data quantity",
        }
    if np.median(policy) < 0.50 and np.median(value) >= 0.50:
        return {
            "label": "policy_gradient_sampling_instability",
            "next_action": "investigate policy-target variance/data quantity",
        }
    if np.median(joint) < 0.50 and arena_complete and not similar:
        return {
            "label": "replay_sampling_instability_confirmed",
            "next_action": "test larger/multi-generation replay window",
        }
    if np.median(joint) >= 0.75 and arena_complete and not similar:
        return {
            "label": "optimizer_step_stable_but_game_effect_unstable",
            "next_action": "investigate nonlinear PUCT sensitivity",
        }
    return {
        "label": "arena_replication_pending",
        "next_action": "complete the preregistered arena matrix before classification",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr191-workdir", type=Path, default=Path("/tmp/azlite_shared_trunk_learning")
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_game_shard_gradient_stability"),
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-game-shard-gradient-stability-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-game-shard-gradient-stability-results.md",
    )
    args = parser.parse_args()
    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact does not match PR191")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, int(manifest["seed"]))
    all_rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(
        manifest["artifact_paths"]["train_source_indexes"], allow_pickle=False
    )
    rows = [all_rows[int(index)] for index in source]
    assignments, shard_manifest = partition(rows)
    (args.workdir / "shards.json").parent.mkdir(parents=True, exist_ok=True)
    (args.workdir / "shards.json").write_text(
        json.dumps(shard_manifest, indent=2, sort_keys=True) + "\n"
    )
    state, optimizer_state = fresh_state(manifest, device)
    batches = {
        name: [
            _batch(rows, indexes, device)
            for indexes in deterministic_batches(indexes, name)
        ]
        for name, indexes in assignments.items()
    }
    raw, updates = {name: [] for name in SHARDS}, {name: [] for name in SHARDS}
    model = new_model(device)
    model.load_state_dict(state)
    model.eval()
    for name in SHARDS:
        for batch in batches[name]:
            raw[name].append(vectors(model, batch))
            updates[name].append(
                virtual_vectors(
                    state,
                    optimizer_state,
                    batch,
                    device,
                    lr=float(manifest["optimizer"]["lr"]),
                    clip=float(manifest["gradient_clip"]),
                )
            )
    candidates = {
        f"joint_{name}_step1": real_candidate(
            state, optimizer_state, batches[name][0], device, manifest
        )
        for name in SHARDS
    }
    repeated = {
        name: real_candidate(
            state, optimizer_state, batches[shard][0], device, manifest
        )
        for name, shard in zip(candidates, SHARDS, strict=True)
    }
    if any(
        state_hash(candidates[name]) != state_hash(repeated[name])
        for name in candidates
    ):
        raise RuntimeError("one-step lane is not deterministic")
    validation = np.load(
        manifest["artifact_paths"]["validation_source_indexes"], allow_pickle=False
    )
    probe, probe_manifest = decoded_validation_manifest(all_rows, validation)
    probe_rows = [all_rows[index] for index in probe_manifest["source_indexes"]]
    probe_hash = stable_hash(probe_manifest)
    all_states = {"C": state, **candidates}
    drift = output_metrics(probe_rows, all_states)
    artifacts = {
        name: export_artifact(candidate, name, args.workdir)
        for name, candidate in candidates.items()
    }
    summary: dict[str, Any] = {
        "schema": "azlite_game_shard_gradient_stability_v1",
        "guardrails": {
            "new_self_play": False,
            "optimizer_steps_per_lane": 1,
            "virtual_steps_chained": False,
            "promotion": False,
        },
        "inputs": {
            "replay_sha256": sha256_file(Path(manifest["replay_path"])),
            "current_weights_sha256": CURRENT_HASH,
        },
        "phase_a": shard_manifest,
        "phase_b": {
            "batches_per_shard": 32,
            "batch_size": 512,
            "pairwise_cosines": pairwise(raw, seed=191),
            "policy_value_conflict": conflict_stability(raw, seed=191),
        },
        "phase_c": {
            "fresh_adam_state": True,
            "gradient_clip": manifest["gradient_clip"],
            "pairwise_cosines": pairwise(updates, seed=192),
            "policy_value_conflict": conflict_stability(updates, seed=192),
        },
        "phase_d": {
            name: {
                "state_sha256": state_hash(candidate),
                "artifact_weights_sha256": sha256_file(
                    artifacts[name] / "weights.json"
                ),
            }
            for name, candidate in candidates.items()
        },
        "phase_e": {
            "probe_manifest": probe_manifest,
            "metrics": drift,
            "pairwise_candidate_agreement": pairwise_output_agreement(
                probe_rows, candidates
            ),
        },
    }
    if args.puct:
        evaluator = {
            "C": arena.ArtifactEvaluator(args.current),
            **{name: arena.ArtifactEvaluator(path) for name, path in artifacts.items()},
        }
        cache_path = args.workdir / "puct_records.jsonl"
        cached = {}
        if cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                cached[(record["name"], record["context"], record["state_hash"])] = (
                    record["record"]
                )
        missing = []
        for context in CONTEXTS:
            for row in probe[:256]:
                for name, value in evaluator.items():
                    key = (name, context, row["state_hash"])
                    if key not in cached:
                        missing.append((key, row, value))
        if missing:
            with cache_path.open("a", encoding="utf-8") as handle:
                for key, row, value in missing:
                    record = puct_record(row, value, key[1], probe_hash)
                    cached[key] = record
                    handle.write(
                        json.dumps(
                            {
                                "name": key[0],
                                "context": key[1],
                                "state_hash": key[2],
                                "record": record,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        puct = {}
        for context in CONTEXTS:
            puct[context] = {}
            for name in candidates:
                pairs = [
                    (
                        cached[("C", context, row["state_hash"])],
                        cached[(name, context, row["state_hash"])],
                    )
                    for row in probe[:256]
                ]
                puct[context][name] = {
                    "move_change_rate": float(
                        np.mean([a["move"] != b["move"] for a, b in pairs])
                    ),
                    "visit_js": float(
                        np.mean(
                            [
                                js(np.asarray([a["visit"]]), np.asarray([b["visit"]]))[
                                    0
                                ]
                                for a, b in pairs
                            ]
                        )
                    ),
                    "true_child_q_ranking_changes": float(
                        np.mean(
                            [
                                a["q_ranking_move_ids"] != b["q_ranking_move_ids"]
                                for a, b in pairs
                            ]
                        )
                    ),
                    "root_value_delta": float(
                        np.mean([b["root_value"] - a["root_value"] for a, b in pairs])
                    ),
                    "visit_margin_delta": float(
                        np.mean([b["margin"] - a["margin"] for a, b in pairs])
                    ),
                }
        summary["phase_f"] = {
            "probe_states": 256,
            "metrics": puct,
            "state_direction_agreement": {
                context: state_direction_agreement(
                    cached, context, candidates, probe[:256]
                )
                for context in CONTEXTS
            },
        }
    if args.arena:
        suite_sha = sha256_file(ARENA_SUITE)
        arena_results = {}
        for context in ("384:256", "1200:1200"):
            challenger_sims, current_sims = (int(value) for value in context.split(":"))
            arena_results[context] = {}
            for name, artifact in artifacts.items():
                candidate_records, control_records = [], []
                for role, challenger, records in (
                    ("candidate", artifact, candidate_records),
                    ("control", args.current, control_records),
                ):
                    for seat in (0, 1):
                        directory = (
                            args.workdir
                            / "arena"
                            / context.replace(":", "_")
                            / name
                            / role
                            / f"starts_{seat}"
                        )
                        directory.mkdir(parents=True, exist_ok=True)
                        records_path = directory / "games.jsonl"
                        existing = (
                            parse_game_jsonl(str(records_path))
                            if records_path.is_file()
                            else []
                        )
                        if len(existing) != 128:
                            run_arena(
                                challenger=str(challenger),
                                current=str(args.current),
                                challenger_sims=challenger_sims,
                                current_sims=current_sims,
                                games=128,
                                seed=191,
                                workers=args.workers,
                                out_json=str(directory / "arena.json"),
                                out_jsonl=str(records_path),
                                opening_prefixes_jsonl=str(ARENA_SUITE),
                                challenger_starts=seat,
                                games_per_opening=1,
                                root_policy_mode="deterministic",
                                root_temperature=0.0,
                                normalize_values=False,
                                c_puct=_context_c_puct(context),
                                tactical_root_bias=0.0,
                                seed_contract=SEED_CONTRACT_VERSION,
                                suite_sha256=suite_sha,
                                seed_ledger_output=str(directory / "seed_ledger.jsonl"),
                            )
                            existing = parse_game_jsonl(str(records_path))
                        records.extend(existing)
                effect = paired_opening_candidate_effect(
                    candidate_records,
                    control_records,
                    bootstrap_samples=10_000,
                    bootstrap_seed=191,
                )
                arena_results[context][name] = {
                    "paired_candidate_effect": effect["paired_candidate_effect"],
                    "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
                }
        summary["phase_g"] = {
            "arena": arena_results,
            "suite_sha256": suite_sha,
            "unique_openings": 128,
            "opening_bootstrap_samples": 10_000,
            "seed_contract": SEED_CONTRACT_VERSION,
        }
    summary["classification"] = classify(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.write_text(
        "# Game-Shard Gradient and One-Step Stability Audit\n\n**Classification:** `"
        + summary["classification"]["label"]
        + "`\n\nFull evidence: `docs/data/alphazero-lite-game-shard-gradient-stability-summary.json`.\n"
    )
    print(
        json.dumps(
            {"classification": summary["classification"], "summary": str(args.summary)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
