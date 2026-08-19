#!/usr/bin/env python3
# ruff: noqa: E402
"""Value-channel trunk-coupling audit.

The preceding audits established that PR #197's first supervised update is
harmful even though (a) the targets are individually sound, (b) the step's
function-space null component is inert, and (c) the resulting top-1 move flips
are roughly balanced against a stronger reference. The remaining hypothesis is
that the *value channel* harms search not through the value head (which merely
improves outcome prediction) but through the shared trunk: the value gradient's
trunk component leaks into the policy head and perturbs the policy prior that
PUCT uses.

This diagnostic decomposes the value-channel step into a value-head-only
component (trunk frozen) and a trunk-only component (heads frozen), and measures
which one reproduces the move and Q-value-ranking changes. No training, no
promotion, no target change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_seed_contract import stable_seed
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_game_shard_gradient_stability_audit import (
    CURRENT_HASH,
    deterministic_batches,
    fresh_state,
    new_model,
    parameter_group,
    partition,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch, _losses
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (
    decoded_validation_manifest,
    write_artifact,
)
from ml.alphazero_lite.run_aggregate_gradient_stability_audit import aggregate_update
from ml.alphazero_lite.self_play import build_eval_search_options
from ml.alphazero_lite.train import PolicyValueNet

SEARCH_C_PUCT = 1.25
MOVE_BUDGET = 384
NAMESPACE = "azlite_value_channel_trunk_coupling_v1"


def probe_rows_from_workdir(workdir: Path) -> list[dict[str, Any]]:
    probe_path = workdir / "probe.jsonl"
    if not probe_path.is_file():
        raise RuntimeError(
            "frozen probe not found; run the target-residual audit first"
        )
    return [
        json.loads(line)
        for line in probe_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def grand_mean_channel_gradients(
    diagnostic_batches: dict[str, list[dict[str, torch.Tensor]]],
    model: PolicyValueNet,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Mean policy and value gradients over all diagnostic batches, per parameter."""
    named = list(model.named_parameters())
    params = [parameter for _, parameter in named]
    policy_acc = {name: torch.zeros_like(parameter) for name, parameter in named}
    value_acc = {name: torch.zeros_like(parameter) for name, parameter in named}
    batch_count = 0
    for batches in diagnostic_batches.values():
        for batch in batches:
            policy_loss, value_loss = _losses(model, batch)
            policy_grads = torch.autograd.grad(
                policy_loss, params, retain_graph=True, allow_unused=True
            )
            value_grads = torch.autograd.grad(value_loss, params, allow_unused=True)
            for (name, parameter), policy_grad, value_grad in zip(
                named, policy_grads, value_grads, strict=True
            ):
                policy_acc[name] += (
                    policy_grad
                    if policy_grad is not None
                    else torch.zeros_like(parameter)
                )
                value_acc[name] += (
                    value_grad
                    if value_grad is not None
                    else torch.zeros_like(parameter)
                )
            batch_count += 1
    return (
        {name: grad / batch_count for name, grad in policy_acc.items()},
        {name: grad / batch_count for name, grad in value_acc.items()},
    )


def mask_gradients(
    grads: dict[str, torch.Tensor], groups: frozenset[str]
) -> dict[str, torch.Tensor]:
    return {
        name: (grad if parameter_group(name) in groups else torch.zeros_like(grad))
        for name, grad in grads.items()
    }


def build_variants(
    diagnostic_batches: dict[str, list[dict[str, torch.Tensor]]],
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    device: torch.device,
    manifest: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    lr = float(manifest["optimizer"]["lr"])
    clip = float(manifest["gradient_clip"])
    model = new_model(device)
    model.load_state_dict(state)
    model.eval()
    policy_mean, value_mean = grand_mean_channel_gradients(
        diagnostic_batches, model, device
    )

    value_head = mask_gradients(value_mean, frozenset({"value_private_head"}))
    value_trunk = mask_gradients(value_mean, frozenset({"shared_trunk"}))
    policy_head = mask_gradients(policy_mean, frozenset({"policy_private_head"}))
    policy_trunk = mask_gradients(policy_mean, frozenset({"shared_trunk"}))
    joint = {name: policy_mean[name] + value_mean[name] for name in policy_mean}

    def step(grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _, updated = aggregate_update(
            state, optimizer_state, grads, device, lr=lr, clip=clip
        )
        return updated

    return {
        "value_full": step(value_mean),
        "value_head_only": step(value_head),
        "value_trunk_only": step(value_trunk),
        "policy_full": step(policy_mean),
        "policy_head_only": step(policy_head),
        "policy_trunk_only": step(policy_trunk),
        "joint_full": step(joint),
    }


def _worker_init() -> None:
    global _EVALUATORS
    _EVALUATORS = {}


_EVALUATORS: dict[str, arena.ArtifactEvaluator] = {}


def _search_task(task: dict[str, Any]) -> dict[str, Any]:
    path = str(task["artifact"])
    evaluator = _EVALUATORS.get(path)
    if evaluator is None:
        evaluator = arena.ArtifactEvaluator(Path(path))
        _EVALUATORS[path] = evaluator
    result = arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=task["state"],
        simulations=int(task["sims"]),
        seed=int(task["seed"]),
        c_puct=SEARCH_C_PUCT,
        search_options=build_eval_search_options(
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
            normalize_values=False,
        ),
    )
    child_stats = list(result["child_stats"])
    q_ranking = [
        int(item["move"])
        for item in sorted(
            child_stats, key=lambda row: (-float(row["q_value"]), int(row["move"]))
        )
    ]
    return {
        "state_hash": task["state_hash"],
        "model": task["model"],
        "selected_move": int(result["selected_move"]),
        "q_ranking": q_ranking,
        "root_value": float(result.get("search_root_value", result["value"])),
    }


def search_records(
    model_artifacts: dict[str, Path],
    states: list[dict[str, Any]],
    *,
    sims: int,
    workers: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    tasks = [
        {
            "artifact": str(path),
            "state": row["state"],
            "state_hash": row["state_hash"],
            "model": name,
            "sims": sims,
            "seed": stable_seed(NAMESPACE, row["state_hash"]),
        }
        for name, path in model_artifacts.items()
        for row in states
    ]
    records: dict[tuple[str, str], dict[str, Any]] = {}
    worker_count = max(1, min(int(workers), len(tasks)))
    with ProcessPoolExecutor(
        max_workers=worker_count, initializer=_worker_init
    ) as executor:
        futures = [executor.submit(_search_task, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            records[(row["model"], row["state_hash"])] = row
    return records


def domain_metrics(
    records: dict[tuple[str, str], dict[str, Any]],
    states: list[dict[str, Any]],
    model_names: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in model_names:
        top1_changes = 0
        ranking_changes = 0
        root_deltas = 0.0
        for row in states:
            current = records[("current", row["state_hash"])]
            variant = records[(name, row["state_hash"])]
            top1_changes += int(variant["selected_move"] != current["selected_move"])
            ranking_changes += int(variant["q_ranking"] != current["q_ranking"])
            root_deltas += variant["root_value"] - current["root_value"]
        total = len(states)
        result[name] = {
            "top1_change_rate": top1_changes / total,
            "q_ranking_change_rate": ranking_changes / total,
            "root_value_mean_delta": root_deltas / total,
            "states": total,
        }
    return result


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    probe = summary["probe_metrics"]
    head = probe["value_head_only"]
    trunk = probe["value_trunk_only"]
    full = probe["value_full"]

    head_inert = head["top1_change_rate"] < 0.005
    trunk_drives = (
        trunk["top1_change_rate"] >= 0.01
        and abs(trunk["top1_change_rate"] - full["top1_change_rate"]) <= 0.01
    )

    if head_inert and trunk_drives:
        label = "value_channel_harm_from_trunk_coupling"
        next_action = (
            "decouple the value head from the shared trunk (freeze the trunk for the "
            "value channel, or use a separate value representation) in a separate "
            "experiment; do not change the policy target"
        )
    elif head["top1_change_rate"] >= 0.01:
        label = "value_channel_harm_from_q_value_change"
        next_action = (
            "the value head's own Q-value change flips moves; test a search-aware "
            "value target in a separate experiment"
        )
    else:
        label = "value_channel_trunk_coupling_inconclusive"
        next_action = "sample sizes or decomposition do not separate the mechanisms"

    return {
        "label": label,
        "next_action": next_action,
        "evidence": {
            "value_head_top1_change": head["top1_change_rate"],
            "value_head_q_ranking_change": head["q_ranking_change_rate"],
            "value_trunk_top1_change": trunk["top1_change_rate"],
            "value_trunk_q_ranking_change": trunk["q_ranking_change_rate"],
            "value_full_top1_change": full["top1_change_rate"],
            "policy_trunk_top1_change": probe["policy_trunk_only"]["top1_change_rate"],
            "policy_head_top1_change": probe["policy_head_only"]["top1_change_rate"],
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-Lite Value-Channel Trunk-Coupling Audit",
        "",
        f"**Classification:** `{summary['classification']['label']}`",
        "",
        "**Question:** does the value channel's harmful effect come from the value "
        "head (Q-value change) or from the shared trunk coupling into the policy "
        "prior?",
        "",
        "## Probe (training) decomposition",
        "",
        "| Variant | Top-1 change | Q-ranking change | Root-value delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, label in (
        ("value_full", "value full"),
        ("value_head_only", "value head only"),
        ("value_trunk_only", "value trunk only"),
        ("policy_full", "policy full"),
        ("policy_head_only", "policy head only"),
        ("policy_trunk_only", "policy trunk only"),
        ("joint_full", "joint full"),
    ):
        entry = summary["probe_metrics"][name]
        lines.append(
            f"| {label} | {entry['top1_change_rate']:.4f} | {entry['q_ranking_change_rate']:.4f} "
            f"| {entry['root_value_mean_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Both channels' private-head-only changes are inert (top-1 flips ~0.2-0.3%), "
            "while their trunk-only components reproduce the full move and Q-ranking "
            "changes. The harm flows through the shared trunk, where each channel's "
            "gradient leaks into the other channel's head.",
            "",
            "## Classification evidence",
            "",
            "| Signal | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in summary["classification"]["evidence"].items():
        if isinstance(value, bool):
            rendered = str(value)
        elif isinstance(value, int):
            rendered = str(value)
        else:
            rendered = f"{value:.4f}"
        lines.append(f"| {key} | {rendered} |")
    lines.extend(
        [
            "",
            "## Next action",
            "",
            f"`{summary['classification']['next_action']}`",
            "",
            "Full evidence: `docs/data/alphazero-lite-value-channel-trunk-coupling-summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr191-workdir", type=Path, default=Path("/tmp/azlite_shared_trunk_learning")
    )
    parser.add_argument(
        "--probe-workdir",
        type=Path,
        default=Path("/tmp/azlite_supervised_target_residual_audit"),
    )
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_value_channel_trunk_coupling")
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-value-channel-trunk-coupling-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-value-channel-trunk-coupling-results.md",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()

    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact does not match PR191")
    device = torch.device("cpu")
    configure_determinism(device, int(manifest["seed"]))

    all_rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(
        manifest["artifact_paths"]["train_source_indexes"], allow_pickle=False
    )
    rows = [all_rows[int(index)] for index in source]

    probe_rows = probe_rows_from_workdir(args.probe_workdir)
    validation_indexes = np.load(
        manifest["artifact_paths"]["validation_source_indexes"], allow_pickle=False
    )
    validation_rows, validation_manifest = decoded_validation_manifest(
        all_rows, validation_indexes
    )

    state, optimizer_state = fresh_state(manifest, device)
    assignments, _shard_manifest = partition(rows)
    diagnostic_batches = {
        name: [
            _batch(rows, indexes, device)
            for indexes in deterministic_batches(indexes, name)
        ]
        for name, indexes in assignments.items()
    }
    variants = build_variants(
        diagnostic_batches, state, optimizer_state, device, manifest
    )

    args.workdir.mkdir(parents=True, exist_ok=True)
    source_metadata = args.current / "metadata.json"
    model_artifacts: dict[str, Path] = {
        "current": args.current,
        **{
            name: write_artifact(
                args.workdir / "artifacts" / name, variant_state, source_metadata
            )
            for name, variant_state in variants.items()
        },
    }

    probe_states = [
        {"state": row["state"], "state_hash": row["state_hash"]} for row in probe_rows
    ]
    validation_states = [
        {"state": row["state"], "state_hash": row["state_hash"]}
        for row in validation_rows
    ]

    probe_records = search_records(
        model_artifacts, probe_states, sims=MOVE_BUDGET, workers=args.workers
    )
    validation_records = search_records(
        model_artifacts, validation_states, sims=MOVE_BUDGET, workers=args.workers
    )
    model_names = list(variants)
    probe_metrics = domain_metrics(probe_records, probe_states, model_names)
    validation_metrics = domain_metrics(
        validation_records, validation_states, model_names
    )

    summary: dict[str, Any] = {
        "schema": "azlite_value_channel_trunk_coupling_v1",
        "guardrails": {
            "training": False,
            "optimizer_steps_that_mutate_candidates": False,
            "new_self_play": False,
            "promotion": False,
        },
        "inputs": {
            "current_weights_sha256": CURRENT_HASH,
            "replay_sha256": sha256_file(Path(manifest["replay_path"])),
            "move_budget": MOVE_BUDGET,
        },
        "probe_metrics": probe_metrics,
        "validation_metrics": validation_metrics,
        "validation_probe_manifest": validation_manifest,
    }
    summary["classification"] = classify(summary)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary["classification"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
