#!/usr/bin/env python3
"""Replay PR #191 exactly and audit within-epoch shared-trunk dynamics.

The only optimizer steps in this program are the original 46 replay steps and
isolated, discarded diagnostic clones.  Virtual steps always begin from an
immutable saved optimizer state and are never chained into a candidate model.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    HIDDEN_SIZES,
    INPUT_ENCODING,
    MODEL_TYPE,
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
    write_fixed_npz,
)
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    _context_c_puct,
    _rank,
    _search_seed,
    _visit_policy,
    decoded_validation_manifest,
    js,
    output_metrics,
    stable_hash,
)
from ml.alphazero_lite.run_terminal_outcome_selfplay_iteration_smoke import (  # noqa: E402
    export_checkpoint,
)
from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    parse_game_jsonl,
    run_arena,
)
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    checkpoint_from_model,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    input_size_for_encoding,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

CURRENT_HASH = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
JOINT_HASH = "df9a7b83763f530f7dfd05a8b5799d8a85e0b4053c126aa725bf1f12f084ea57"
TRUNK_PREFIXES = ("input_layer.", "residual_layers.")
FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
SEARCH_CONTEXTS = ("384:256", "768:768", "1200:1200")
ARENA_SUITE = Path(
    "/tmp/azlite_shared_trunk_learning/arena_vs_current/temp_0_0/seed_42/artifact/equal_high/starts_0/opening_suite.jsonl"
)


def snapshot_steps(batch_count: int) -> list[int]:
    """Return unique real optimizer boundaries for the protocol fractions."""
    if batch_count < 1:
        raise ValueError("batch plan must contain at least one batch")
    # Half-up avoids Python's banker's rounding at the 34.5-step boundary.
    return sorted({math.floor(batch_count * fraction + 0.5) for fraction in FRACTIONS})


def trunk_group(name: str) -> str | None:
    """Map a parameter name to the requested trunk aggregation group."""
    if name.startswith("input_layer."):
        return "input_layer"
    if name.startswith("residual_layers."):
        return f"residual_block_{name.split('.')[1]}"
    return None


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return a finite cosine for flattened diagnostic vectors."""
    return float(
        torch.dot(left, right)
        / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right) + 1e-20)
    )


def vector_metrics(vectors: dict[str, torch.Tensor]) -> dict[str, float]:
    """Summarize the three requested gradient or update vectors."""
    policy, value, joint = (vectors[key] for key in ("policy", "value", "joint"))
    return {
        "policy_norm": float(torch.linalg.vector_norm(policy)),
        "value_norm": float(torch.linalg.vector_norm(value)),
        "joint_norm": float(torch.linalg.vector_norm(joint)),
        "cosine_policy_value": cosine(policy, value),
        "cosine_policy_joint": cosine(policy, joint),
        "cosine_value_joint": cosine(value, joint),
    }


def bootstrap_summary(
    samples: list[dict[str, float]], *, seed: int
) -> dict[str, dict[str, float]]:
    """Bootstrap batch means for every numeric diagnostic statistic."""
    if not samples:
        return {}
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(samples), size=(10_000, len(samples)))
    result = {}
    for key in samples[0]:
        values = np.asarray([sample[key] for sample in samples], dtype=np.float64)
        draws = values[indexes].mean(axis=1)
        result[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "lower_95": float(np.quantile(draws, 0.025)),
            "upper_95": float(np.quantile(draws, 0.975)),
        }
    return result


def _new_model(device: torch.device) -> PolicyValueNet:
    model = PolicyValueNet(
        HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding(INPUT_ENCODING)
    )
    return model.to(device)


def _trunk_parameters(model: PolicyValueNet) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if trunk_group(name) is not None
    ]


def _losses(
    model: PolicyValueNet, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, prediction = model(batch["x"])
    policy = compute_policy_cross_entropy(
        logits.masked_fill(batch["mask"] <= 0, -1e9), batch["p"]
    ).mean()
    value = compute_value_loss_vector(
        prediction, batch["v"], value_loss="huber", huber_delta=1.0
    ).mean()
    return policy, 0.6 * value


def _batch(
    rows: list[dict[str, Any]], indexes: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    selected = [rows[int(index)] for index in indexes[indexes >= 0]]
    x = np.asarray([row["state"] for row in selected], dtype=np.float32)
    return {
        "x": torch.from_numpy(x).to(device),
        "p": torch.tensor(
            np.asarray([row["policy"] for row in selected], dtype=np.float32),
            device=device,
        ),
        "v": torch.tensor(
            np.asarray([row["value"] for row in selected], dtype=np.float32).reshape(
                -1, 1
            ),
            device=device,
        ),
        "mask": torch.from_numpy(legal_mask_matrix_for_encoded_states(x)).to(device),
    }


def _vectors_from_gradients(
    model: PolicyValueNet, loss: torch.Tensor
) -> dict[str, torch.Tensor]:
    named = _trunk_parameters(model)
    gradients = torch.autograd.grad(
        loss, [parameter for _, parameter in named], retain_graph=True
    )
    grouped: dict[str, list[torch.Tensor]] = {"global": []}
    for (name, _parameter), gradient in zip(named, gradients, strict=True):
        grouped["global"].append(gradient.detach().reshape(-1))
        grouped.setdefault(trunk_group(name) or "global", []).append(
            gradient.detach().reshape(-1)
        )
    return {name: torch.cat(parts) for name, parts in grouped.items()}


def raw_gradient_sample(
    model: PolicyValueNet, batch: dict[str, torch.Tensor]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, torch.Tensor]]]:
    """Calculate policy, weighted-value, and joint trunk gradients without stepping."""
    policy_loss, value_loss = _losses(model, batch)
    policy = _vectors_from_gradients(model, policy_loss)
    value = _vectors_from_gradients(model, value_loss)
    joint = {group: policy[group] + value[group] for group in policy}
    vectors = {
        group: {"policy": policy[group], "value": value[group], "joint": joint[group]}
        for group in policy
    }
    return {group: vector_metrics(vector) for group, vector in vectors.items()}, vectors


def virtual_update_vectors(
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    lr: float,
    clip: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, torch.Tensor]]]:
    """Measure independent Adam updates from one identical pre-step state."""
    all_updates: dict[str, dict[str, torch.Tensor]] = {}
    for loss_name in ("policy", "value", "joint"):
        model = _new_model(device)
        model.load_state_dict(state)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        before = {
            name: parameter.detach().clone()
            for name, parameter in _trunk_parameters(model)
        }
        policy_loss, value_loss = _losses(model, batch)
        loss = {
            "policy": policy_loss,
            "value": value_loss,
            "joint": policy_loss + value_loss,
        }[loss_name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        grouped: dict[str, list[torch.Tensor]] = {"global": []}
        for name, parameter in _trunk_parameters(model):
            delta = (parameter.detach() - before[name]).reshape(-1)
            grouped["global"].append(delta)
            grouped.setdefault(trunk_group(name) or "global", []).append(delta)
        all_updates[loss_name] = {
            group: torch.cat(parts) for group, parts in grouped.items()
        }
    vectors = {
        group: {kind: updates[group] for kind, updates in all_updates.items()}
        for group in all_updates["joint"]
    }
    return {group: vector_metrics(vector) for group, vector in vectors.items()}, vectors


def audit_snapshot(
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    lr: float,
    clip: float,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    """Audit fixed batches and retain only their mean virtual directions in memory."""
    raw: dict[str, list[dict[str, float]]] = {}
    updates: dict[str, list[dict[str, float]]] = {}
    update_vectors: dict[str, dict[str, list[torch.Tensor]]] = {}
    model = _new_model(device)
    model.load_state_dict(state)
    model.eval()
    for batch in batches:
        raw_metrics, _raw_vectors = raw_gradient_sample(model, batch)
        update_metrics, vectors = virtual_update_vectors(
            state, optimizer_state, batch, device, lr=lr, clip=clip
        )
        for group, metrics in raw_metrics.items():
            raw.setdefault(group, []).append(metrics)
        for group, metrics in update_metrics.items():
            updates.setdefault(group, []).append(metrics)
        for group, by_kind in vectors.items():
            for kind, vector in by_kind.items():
                update_vectors.setdefault(group, {}).setdefault(kind, []).append(
                    vector.cpu()
                )
    means = {
        group: {kind: torch.stack(values).mean(0) for kind, values in by_kind.items()}
        for group, by_kind in update_vectors.items()
    }
    return {
        "raw_gradients": {
            group: bootstrap_summary(samples, seed=191)
            for group, samples in raw.items()
        },
        "virtual_updates": {
            group: bootstrap_summary(samples, seed=191)
            for group, samples in updates.items()
        },
        "fixed_batches": len(batches),
        "virtual_steps_chained": False,
    }, means


def _save_snapshot(
    path: Path, model: PolicyValueNet, optimizer: torch.optim.Optimizer
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "optimizer": optimizer_state}, path)
    return state, optimizer_state


def replay_with_snapshots(
    manifest: dict[str, Any], workdir: Path, device: torch.device
) -> tuple[
    dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]], list[dict[str, Any]]
]:
    """Perform the exact joint-trunk replay, saving only optimizer boundaries."""
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source, plan = (
        np.load(paths["train_source_indexes"], allow_pickle=False),
        np.load(paths["batch_indexes"], allow_pickle=False),
    )
    selected = [rows[int(index)] for index in source]
    batches = [_batch(selected, indexes, device) for indexes in plan]
    model = _new_model(device)
    load_checkpoint_into_model(model, paths["initialization_checkpoint"])
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(manifest["optimizer"]["lr"]),
        weight_decay=float(manifest["optimizer"].get("weight_decay", 0.0)),
    )
    wanted = snapshot_steps(len(plan))
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]] = {}
    snapshots[0] = _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)
    for step, batch in enumerate(batches, 1):
        policy, value = _losses(model, batch)
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(manifest["gradient_clip"])
        )
        optimizer.step()
        if step in wanted:
            snapshots[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )
    if set(snapshots) != set(wanted):
        raise RuntimeError("failed to capture every required optimizer boundary")
    return snapshots, batches


def _weights_hash(state: dict[str, torch.Tensor], workdir: Path) -> str:
    model = _new_model(torch.device("cpu"))
    model.load_state_dict(state)
    checkpoint = workdir / "final_replay.npz"
    write_fixed_npz(checkpoint, checkpoint_from_model(model))
    artifact = workdir / "final_replay_artifact"
    export_checkpoint(
        checkpoint_path=checkpoint,
        out_dir=artifact,
        version="pr191_joint_replay",
        policy_loss=0.0,
        value_loss=0.0,
    )
    return sha256_file(artifact / "weights.json")


def output_drift(
    rows: list[dict[str, Any]],
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
) -> dict[str, Any]:
    """Evaluate all real replay boundaries on the immutable PR192 512-state probe."""
    states = {"C": snapshots[0][0]}
    states.update({str(step): state for step, (state, _optimizer) in snapshots.items()})
    metrics = output_metrics(rows, states)
    result = {}
    for step in snapshots:
        item = metrics[str(step)]
        result[str(step)] = {
            "policy": {
                "legal_kl_from_current": item["policy"]["kl_from_current"],
                "replay_teacher_cross_entropy": item["policy"][
                    "legal_cross_entropy_to_replay_teacher"
                ],
                "replay_teacher_js": item["policy"]["js_to_replay_teacher"],
                "top1_change_from_current": 1.0
                - item["policy"]["top1_agreement_with_current"],
            },
            "value": {
                "canonical_outcome_mae": item["value"]["mae"],
                "huber_loss": item["value"]["huber_loss_to_canonical_outcome"],
                "sign_accuracy": item["value"]["sign_accuracy"],
                "pairwise_concordance": item["value"]["pairwise_concordance"],
            },
        }
    return result


def export_snapshot_artifacts(
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]], workdir: Path
) -> dict[int, Path]:
    """Export immutable diagnostic artifacts; they are never candidate recipes."""
    artifacts: dict[int, Path] = {}
    for step, (state, _optimizer) in snapshots.items():
        directory = workdir / "snapshot_artifacts" / f"step_{step:04d}"
        artifact = directory / "artifact"
        if not (artifact / "weights.json").is_file():
            model = _new_model(torch.device("cpu"))
            model.load_state_dict(state)
            checkpoint = directory / "checkpoint.npz"
            directory.mkdir(parents=True, exist_ok=True)
            write_fixed_npz(checkpoint, checkpoint_from_model(model))
            export_checkpoint(
                checkpoint_path=checkpoint,
                out_dir=artifact,
                version=f"pr191_dynamics_step_{step:04d}",
                policy_loss=0.0,
                value_loss=0.0,
            )
            metadata_path = artifact / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["diagnostic_only"] = True
            metadata["promotion_forbidden"] = True
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        artifacts[step] = artifact
    return artifacts


def _search_row(
    probe_row: dict[str, Any],
    evaluator: arena.ArtifactEvaluator,
    context: str,
    manifest_hash: str,
) -> dict[str, Any]:
    """Run one canonical deterministic zero-Dirichlet root search."""
    seed, seed_context_hash = _search_seed(probe_row, manifest_hash, 191)
    result = arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=probe_row["state"],
        simulations=int(context.split(":", maxsplit=1)[0]),
        seed=seed,
        c_puct=_context_c_puct(context),
        search_options=build_eval_search_options(
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
            normalize_values=False,
        ),
    )
    child_stats = list(result["child_stats"])
    selected_move = int(result["selected_move"])
    visits = sorted((int(row["visits"]) for row in child_stats), reverse=True)
    return {
        "state_hash": probe_row["state_hash"],
        "manifest_index": probe_row["manifest_index"],
        "context": context,
        "search_seed": seed,
        "seed_context_hash": seed_context_hash,
        "selected_move": selected_move,
        "visit_policy": _visit_policy(child_stats),
        "root_value": float(result.get("search_root_value", result["value"])),
        "selected_child_q_rank": _rank(selected_move, child_stats, "q_value"),
        "top1_top2_visit_margin": int(visits[0] - visits[1])
        if len(visits) > 1
        else int(visits[0]),
    }


def puct_trajectory(
    probe: list[dict[str, Any]],
    artifacts: dict[int, Path],
    workdir: Path,
    manifest_hash: str,
    contexts: tuple[str, ...] = SEARCH_CONTEXTS,
) -> dict[str, Any]:
    """Cache snapshot-versus-current PUCT drift for the preregistered budgets."""
    cache = workdir / "phase_f_puct_records.jsonl"
    records: dict[tuple[int, str, str], dict[str, Any]] = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            records[(int(row["step"]), str(row["context"]), str(row["state_hash"]))] = (
                row
            )
    evaluators = {
        step: arena.ArtifactEvaluator(path) for step, path in artifacts.items()
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a", encoding="utf-8") as stream:
        for step in sorted(artifacts):
            for context in contexts:
                for probe_row in probe:
                    key = (step, context, probe_row["state_hash"])
                    if key in records:
                        continue
                    row = _search_row(
                        probe_row, evaluators[step], context, manifest_hash
                    )
                    row["step"] = step
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    records[key] = row
    summary: dict[str, Any] = {}
    for step in sorted(artifacts):
        for context in contexts:
            pairs = [
                (
                    records[(0, context, row["state_hash"])],
                    records[(step, context, row["state_hash"])],
                )
                for row in probe
            ]
            summary.setdefault(str(step), {})[context] = {
                "states": len(pairs),
                "selected_move_change_rate": float(
                    np.mean(
                        [
                            left["selected_move"] != right["selected_move"]
                            for left, right in pairs
                        ]
                    )
                ),
                "visit_js": float(
                    np.mean(
                        [
                            js(
                                np.asarray([left["visit_policy"]]),
                                np.asarray([right["visit_policy"]]),
                            )[0]
                            for left, right in pairs
                        ]
                    )
                ),
                "root_value_delta": float(
                    np.mean(
                        [
                            right["root_value"] - left["root_value"]
                            for left, right in pairs
                        ]
                    )
                ),
                "child_q_rank_change": float(
                    np.mean(
                        [
                            right["selected_child_q_rank"]
                            - left["selected_child_q_rank"]
                            for left, right in pairs
                        ]
                    )
                ),
                "visit_margin": float(
                    np.mean(
                        [
                            right["top1_top2_visit_margin"]
                            - left["top1_top2_visit_margin"]
                            for left, right in pairs
                        ]
                    )
                ),
            }
    return {
        "records_path": str(cache),
        "settings": {
            "seed_contract": "azlite_eval_seed_v2",
            "tactical_root_bias": 0.0,
            "root_policy_mode": "deterministic",
            "dirichlet": 0.0,
        },
        "metrics": summary,
    }


def _suite_provenance(path: Path) -> dict[str, Any]:
    """Verify the existing canonical suite without constructing another opening set."""
    metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))[
        "cache_manifest"
    ]
    source = Path(metadata["suite_path"])
    if (
        len(path.read_text(encoding="utf-8").splitlines()) != 128
        or sha256_file(source) != metadata["suite_sha256"]
    ):
        raise RuntimeError(
            "canonical arena suite provenance does not match its persisted source"
        )
    return {
        "suite_path": str(path),
        "suite_sha256": metadata["suite_sha256"],
        "unique_openings": 128,
    }


def sparse_arena(
    artifacts: dict[int, Path], current: Path, workdir: Path, *, workers: int
) -> dict[str, Any]:
    """Run only the preregistered before/first-conflict/final paired checkpoints."""
    suite = _suite_provenance(ARENA_SUITE)
    results: dict[str, Any] = {"suite": suite, "checkpoints": [0, 5, 46], "metrics": {}}
    for step in (5, 46):
        for context in ("384:256", "1200:1200"):
            challenger_sims, current_sims = (int(value) for value in context.split(":"))
            candidate_records: list[dict[str, Any]] = []
            control_records: list[dict[str, Any]] = []
            for role, challenger, records in (
                ("candidate", artifacts[step], candidate_records),
                ("current_control", current, control_records),
            ):
                for seat in (0, 1):
                    directory = (
                        workdir
                        / "sparse_arena"
                        / f"step_{step:04d}"
                        / context.replace(":", "_")
                        / role
                        / f"starts_{seat}"
                    )
                    records_path = directory / "arena.jsonl"
                    expected = 256
                    if (
                        not records_path.is_file()
                        or len(parse_game_jsonl(str(records_path))) != expected
                    ):
                        directory.mkdir(parents=True, exist_ok=True)
                        run_arena(
                            challenger=str(challenger),
                            current=str(current),
                            challenger_sims=challenger_sims,
                            current_sims=current_sims,
                            games=expected,
                            seed=42,
                            workers=workers,
                            out_json=str(directory / "arena.json"),
                            out_jsonl=str(records_path),
                            opening_prefixes_jsonl=str(ARENA_SUITE),
                            challenger_starts=seat,
                            games_per_opening=2,
                            root_policy_mode="deterministic",
                            root_temperature=0.0,
                            normalize_values=False,
                            c_puct=_context_c_puct(context),
                            tactical_root_bias=0.0,
                            suite_sha256=suite["suite_sha256"],
                            seed_ledger_output=str(directory / "seed_ledger.jsonl"),
                        )
                    records.extend(parse_game_jsonl(str(records_path)))
            effect = paired_opening_candidate_effect(
                candidate_records,
                control_records,
                bootstrap_samples=10_000,
                bootstrap_seed=42,
            )
            results["metrics"].setdefault(str(step), {})[context] = {
                "paired_candidate_effect": effect["paired_candidate_effect"],
                "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
                "orientation": "snapshot_minus_current",
            }
    results["metrics"]["0"] = {
        context: {
            "paired_candidate_effect": 0.0,
            "opening_bootstrap_ci": {"lower_95": 0.0, "upper_95": 0.0},
            "orientation": "current_minus_current_identity",
        }
        for context in ("384:256", "1200:1200")
    }
    return results


def movement_metrics(
    current: dict[str, torch.Tensor],
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    local: dict[int, dict[str, dict[str, torch.Tensor]]],
) -> dict[str, Any]:
    """Relate each accumulated real trunk delta to local and cumulative updates."""
    result: dict[str, Any] = {}
    running: dict[str, dict[str, torch.Tensor]] = {}
    for step in sorted(snapshots):
        state, _optimizer = snapshots[step]
        for group, update in local[step].items():
            running.setdefault(
                group,
                {kind: torch.zeros_like(vector) for kind, vector in update.items()},
            )
            for kind, vector in update.items():
                running[group][kind] += vector
        groups: dict[str, Any] = {}
        for group, directions in local[step].items():
            names = [
                (name, value)
                for name, value in state.items()
                if trunk_group(name) == (None if group == "global" else group)
            ]
            if group == "global":
                names = [
                    (name, value)
                    for name, value in state.items()
                    if trunk_group(name) is not None
                ]
            delta = torch.cat(
                [(value - current[name]).reshape(-1) for name, value in names]
            )
            base = torch.cat([current[name].reshape(-1) for name, _value in names])
            groups[group] = {
                "relative_l2": float(
                    torch.linalg.vector_norm(delta)
                    / (torch.linalg.vector_norm(base) + 1e-20)
                ),
                "local_virtual_cosines": {
                    kind: cosine(delta, vector) for kind, vector in directions.items()
                },
                "average_virtual_cosines": {
                    kind: cosine(delta, vector)
                    for kind, vector in running[group].items()
                },
            }
        result[str(step)] = groups
    return result


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the prespecified classifications without selecting a treatment."""
    ordered = [summary["snapshots"][str(step)] for step in summary["snapshot_steps"]]
    raw_high = all(
        item["audit"]["raw_gradients"]["global"]["policy_norm"]["median"]
        / item["audit"]["raw_gradients"]["global"]["value_norm"]["median"]
        >= 3
        for item in ordered
    )
    update_ratios = [
        item["audit"]["virtual_updates"]["global"]["policy_norm"]["median"]
        / item["audit"]["virtual_updates"]["global"]["value_norm"]["median"]
        for item in ordered
    ]
    joint_policy = [
        item["audit"]["virtual_updates"]["global"]["cosine_policy_joint"]["mean"]
        for item in ordered
    ]
    movement = summary["movement"]
    policy_aligned = all(
        value["global"]["average_virtual_cosines"]["policy"]
        > value["global"]["average_virtual_cosines"]["value"]
        for value in movement.values()
        if value["global"]["relative_l2"] > 0
    )
    conflict_steps = [
        step
        for step, item in zip(summary["snapshot_steps"], ordered, strict=True)
        if item["audit"]["raw_gradients"]["global"]["cosine_policy_value"]["upper_95"]
        < 0
    ]
    initialization_ci = ordered[0]["audit"]["raw_gradients"]["global"][
        "cosine_policy_value"
    ]
    later_conflict = [step in conflict_steps for step in summary["snapshot_steps"][1:]]
    confirmed = initialization_ci["lower_95"] <= 0 <= initialization_ci[
        "upper_95"
    ] and any(
        left and right
        for left, right in zip(later_conflict, later_conflict[1:], strict=False)
    )
    labels = []
    if (
        raw_high
        and all(value >= 3 for value in update_ratios)
        and all(value >= 0.90 for value in joint_policy)
        and policy_aligned
    ):
        labels.append("optimizer_policy_dominance_confirmed")
    if raw_high and (
        any(value < 2 for value in update_ratios)
        or not all(value >= 0.90 for value in joint_policy)
    ):
        labels.append("raw_gradient_dominance_not_optimizer_dominance")
    if confirmed:
        labels.append("gradient_conflict_emerges_during_training")
    next_action = "no_prespecified_training_change_justified"
    arena_metrics = summary.get("phase_g_sparse_arena", {}).get("metrics", {})
    if arena_metrics:

        def harmful(effect: dict[str, Any]) -> bool:
            return (
                effect["paired_candidate_effect"] <= -0.03
                and effect["opening_bootstrap_ci"]["upper_95"] < 0
            )

        before_neutral = all(
            value["opening_bootstrap_ci"]["lower_95"]
            <= 0
            <= value["opening_bootstrap_ci"]["upper_95"]
            for value in arena_metrics["0"].values()
        )
        after_harmful = any(
            harmful(value) for value in arena_metrics.get("5", {}).values()
        )
        if before_neutral and after_harmful:
            labels.append("search_harm_tracks_gradient_conflict")
            next_action = "compare_normal_joint_trunk_vs_one_prespecified_conflict_mitigated_trunk_gradient"
    if "raw_gradient_dominance_not_optimizer_dominance" in labels:
        next_action = (
            "retire_the_7_94x_raw_gradient_argument_before_changing_loss_weights; "
            + next_action
        )
    return {
        "classifications": labels,
        "conflict_steps": conflict_steps,
        "next_action": next_action,
    }


def markdown(summary: dict[str, Any]) -> str:
    """Render a compact committed record while retaining full data in JSON."""
    lines = [
        "# AlphaZero-Lite Optimizer-Aware Trunk Dynamics Audit",
        "",
        f"**Classification:** `{', '.join(summary['classification']['classifications']) or 'none'}`",
        "",
        f"- PR191 replay hash: `{summary['phase_a']['final_weights_sha256']}`",
        f"- expected joint hash matched: `{summary['phase_a']['reproduced_exactly']}`",
        f"- snapshots: `{summary['snapshot_steps']}`",
        f"- confirmed conflict steps: `{summary['classification']['conflict_steps']}`",
        f"- next action: `{summary['classification']['next_action']}`",
        "",
        "## PUCT Trajectory",
        "",
    ]
    if "phase_f_puct_trajectory" in summary:
        lines.extend(
            [
                "| Step | Context | Move change | Visit JS | Root-value delta | Q-rank change | Visit-margin delta |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for step in summary["snapshot_steps"]:
            for context, metrics in summary["phase_f_puct_trajectory"]["metrics"][
                str(step)
            ].items():
                lines.append(
                    f"| {step} | {context} | {metrics['selected_move_change_rate']:.4f} | "
                    f"{metrics['visit_js']:.4f} | {metrics['root_value_delta']:+.4f} | "
                    f"{metrics['child_q_rank_change']:+.4f} | {metrics['visit_margin']:+.2f} |"
                )
    if "phase_g_sparse_arena" in summary:
        lines.extend(
            [
                "",
                "## Sparse Arena",
                "",
                "| Step | Context | Paired effect | 95% CI |",
                "| ---: | --- | ---: | --- |",
            ]
        )
        for step, contexts in summary["phase_g_sparse_arena"]["metrics"].items():
            for context, metrics in contexts.items():
                ci = metrics["opening_bootstrap_ci"]
                lines.append(
                    f"| {step} | {context} | {metrics['paired_candidate_effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
                )
    lines.extend(
        [
            "",
            "Full fixed-batch gradients, isolated Adam virtual updates, real trajectory alignment, frozen-probe data, and PUCT records are in `docs/data/alphazero-lite-optimizer-aware-trunk-dynamics-summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_optimizer_aware_trunk_dynamics"),
    )
    parser.add_argument(
        "--pr191-workdir", type=Path, default=Path("/tmp/azlite_shared_trunk_learning")
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-optimizer-aware-trunk-dynamics-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-optimizer-aware-trunk-dynamics-results.md",
    )
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--arena-workers", type=int, default=24)
    args = parser.parse_args()
    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact does not match the PR191 initialization")
    configure_determinism(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        int(manifest["seed"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshots, train_batches = replay_with_snapshots(manifest, args.workdir, device)
    final_hash = _weights_hash(snapshots[max(snapshots)][0], args.workdir)
    if final_hash != JOINT_HASH:
        raise RuntimeError(
            f"PR191 replay hash mismatch: expected {JOINT_HASH}, got {final_hash}"
        )
    diagnostics = train_batches[:32]
    audit_by_step, update_directions = {}, {}
    for step, (state, optimizer_state) in snapshots.items():
        audit_by_step[step], update_directions[step] = audit_snapshot(
            state,
            optimizer_state,
            diagnostics,
            device,
            lr=float(manifest["optimizer"]["lr"]),
            clip=float(manifest["gradient_clip"]),
        )
    current_state = snapshots[0][0]
    rows = read_jsonl(Path(manifest["replay_path"]))
    validation_indexes = np.load(
        Path(manifest["artifact_paths"]["validation_source_indexes"]),
        allow_pickle=False,
    )
    _probe, probe_manifest = decoded_validation_manifest(rows, validation_indexes)
    probe_manifest["validation_source_indexes_sha256"] = sha256_file(
        Path(manifest["artifact_paths"]["validation_source_indexes"])
    )
    probe_manifest["replay_sha256"] = sha256_file(Path(manifest["replay_path"]))
    probe_manifest["manifest_sha256"] = stable_hash(probe_manifest)
    probe_rows = [rows[index] for index in probe_manifest["source_indexes"]]
    summary: dict[str, Any] = {
        "schema": "azlite_optimizer_aware_trunk_dynamics_v1",
        "guardrails": {
            "new_replay": False,
            "candidate_recipe": False,
            "promotion": False,
            "virtual_steps_chained": False,
        },
        "inputs": {
            "training_manifest": str(args.pr191_workdir / "training_manifest.json"),
            "current_weights_sha256": CURRENT_HASH,
            "expected_joint_weights_sha256": JOINT_HASH,
        },
        "phase_a": {
            "batch_count": len(train_batches),
            "final_weights_sha256": final_hash,
            "reproduced_exactly": True,
            "optimizer": manifest["optimizer"],
            "gradient_clip": manifest["gradient_clip"],
            "scheduler": "none",
        },
        "snapshot_steps": list(snapshots),
        "snapshots": {
            str(step): {
                "training_fraction": step / len(train_batches),
                "audit": audit_by_step[step],
            }
            for step in snapshots
        },
        "movement": movement_metrics(current_state, snapshots, update_directions),
        "phase_e_output_drift": {
            "probe_manifest": probe_manifest,
            "metrics": output_drift(probe_rows, snapshots),
        },
    }
    if args.puct:
        artifacts = export_snapshot_artifacts(snapshots, args.workdir)
        summary["phase_f_puct_trajectory"] = puct_trajectory(
            _probe, artifacts, args.workdir, probe_manifest["manifest_sha256"]
        )
    if args.arena:
        artifacts = export_snapshot_artifacts(snapshots, args.workdir)
        summary["phase_g_sparse_arena"] = sparse_arena(
            artifacts, args.current, args.workdir, workers=args.arena_workers
        )
    summary["classification"] = classify(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {"reproduced_exactly": True, "classification": summary["classification"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
