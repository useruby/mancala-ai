#!/usr/bin/env python3
# ruff: noqa: E402
"""Search-aware distillation channel attribution.

The two preceding audits established that PR #197's first supervised update is
harmful even though (a) the policy and value targets are individually sound and
(b) the harmful step's function-space null component is inert. The remaining
hypothesis is that the small function-space output change degrades search
outcomes: the distillation step moves the model's policy/value in a direction
that flips deterministic PUCT moves *away* from the move a stronger search
would select.

This diagnostic decomposes the harmful step into its policy-channel and
value-channel components and measures, on the frozen training probe and the
held-out validation probe, whether each channel's move flips are aligned with
or opposed to a stronger D1200 reference. No training, no promotion, no target
change.
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
    partition,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch, _losses
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (
    decoded_validation_manifest,
    write_artifact,
)
from ml.alphazero_lite.run_aggregate_gradient_stability_audit import aggregate_update
from ml.alphazero_lite.self_play import build_eval_search_options

SEARCH_C_PUCT = 1.25
MOVE_BUDGET = 384
REFERENCE_BUDGET = 1200
NAMESPACE = "azlite_search_aware_distillation_channel_v1"


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


def channel_directions(
    diagnostic_batches: dict[str, list[dict[str, torch.Tensor]]],
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    device: torch.device,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Grand-mean policy/value/joint gradients and their one-step Adam states."""
    lr = float(manifest["optimizer"]["lr"])
    clip = float(manifest["gradient_clip"])
    model = new_model(device)
    model.load_state_dict(state)
    model.eval()
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
    policy_mean = {name: grad / batch_count for name, grad in policy_acc.items()}
    value_mean = {name: grad / batch_count for name, grad in value_acc.items()}
    joint_mean = {name: policy_mean[name] + value_mean[name] for name in policy_mean}
    _, policy_state = aggregate_update(
        state, optimizer_state, policy_mean, device, lr=lr, clip=clip
    )
    _, value_state = aggregate_update(
        state, optimizer_state, value_mean, device, lr=lr, clip=clip
    )
    _, joint_state = aggregate_update(
        state, optimizer_state, joint_mean, device, lr=lr, clip=clip
    )
    return {
        "policy_state": policy_state,
        "value_state": value_state,
        "joint_state": joint_state,
        "policy_grad_norm": float(
            np.sqrt(sum(float(torch.sum(grad**2)) for grad in policy_mean.values()))
        ),
        "value_grad_norm": float(
            np.sqrt(sum(float(torch.sum(grad**2)) for grad in value_mean.values()))
        ),
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
    return {
        "state_hash": task["state_hash"],
        "model": task["model"],
        "sims": int(task["sims"]),
        "selected_move": int(result["selected_move"]),
    }


def search_moves(
    model_artifacts: dict[str, Path],
    states: list[dict[str, Any]],
    *,
    sims: int,
    workers: int,
) -> dict[tuple[str, str], int]:
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
    records: dict[tuple[str, str], int] = {}
    worker_count = max(1, min(int(workers), len(tasks)))
    with ProcessPoolExecutor(
        max_workers=worker_count, initializer=_worker_init
    ) as executor:
        futures = [executor.submit(_search_task, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            records[(row["model"], row["state_hash"])] = row["selected_move"]
    return records


def reference_alignment(
    records: dict[tuple[str, str], int],
    model: str,
    reference_records: dict[str, int],
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    aligned = 0
    flips_toward = 0
    flips_away = 0
    flips = 0
    for row in states:
        reference = reference_records[row["state_hash"]]
        current_move = records[("current", row["state_hash"])]
        variant_move = records[(model, row["state_hash"])]
        if variant_move == reference:
            aligned += 1
        if variant_move != current_move:
            flips += 1
            if variant_move == reference:
                flips_toward += 1
            elif current_move == reference:
                flips_away += 1
    total = len(states)
    return {
        "states": total,
        "reference_alignment": aligned / total,
        "move_change_rate": flips / total,
        "flips_toward_reference": flips_toward / total,
        "flips_away_from_reference": flips_away / total,
        "net_reference_delta": (flips_toward - flips_away) / total,
    }


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    probe = summary["probe_metrics"]
    validation = summary["validation_metrics"]

    def misaligned(name: str) -> bool:
        probe_entry = probe[name]
        validation_entry = validation[name]
        return (
            probe_entry["move_change_rate"] >= 0.01
            and validation_entry["move_change_rate"] >= 0.01
            and probe_entry["net_reference_delta"] < -0.005
            and validation_entry["net_reference_delta"] < -0.005
        )

    policy_misaligned = misaligned("policy")
    value_misaligned = misaligned("value")

    if policy_misaligned and value_misaligned:
        label = "both_channels_search_misaligned"
        next_action = (
            "test search-aware targets for both policy and value channels in "
            "separate experiments (do not change both at once)"
        )
    elif policy_misaligned:
        label = "policy_channel_search_misaligned"
        next_action = (
            "test a search-aware policy target (stronger/deterministic-search "
            "policy) in a separate experiment; leave the value target unchanged"
        )
    elif value_misaligned:
        label = "value_channel_search_misaligned"
        next_action = (
            "test a search-aware value target in a separate experiment; leave the "
            "policy target unchanged"
        )
    else:
        label = "search_aware_top1_misalignment_not_confirmed"
        next_action = (
            "the harmful step flips deterministic moves roughly balanced toward and "
            "away from a stronger D1200 reference; investigate the value/Q-value "
            "channel's effect on PUCT root rankings or treat distillation as saturated"
        )

    return {
        "label": label,
        "next_action": next_action,
        "evidence": {
            "joint_probe_move_change": probe["joint"]["move_change_rate"],
            "policy_probe_move_change": probe["policy"]["move_change_rate"],
            "value_probe_move_change": probe["value"]["move_change_rate"],
            "policy_probe_net_reference_delta": probe["policy"]["net_reference_delta"],
            "value_probe_net_reference_delta": probe["value"]["net_reference_delta"],
            "validation_joint_move_change": validation["joint"]["move_change_rate"],
            "validation_policy_move_change": validation["policy"]["move_change_rate"],
            "validation_value_move_change": validation["value"]["move_change_rate"],
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-Lite Search-Aware Distillation Channel Audit",
        "",
        f"**Classification:** `{summary['classification']['label']}`",
        "",
        "**Question:** does the harmful step's policy channel or value channel "
        "drive deterministic-search move flips, and do those flips move toward "
        "or away from a stronger D1200 reference?",
        "",
        f"- policy/value grand-mean gradient norm ratio: "
        f"{summary['gradient_norms']['policy_grad_norm'] / max(summary['gradient_norms']['value_grad_norm'], 1e-20):.2f}",
        "",
        "## Probe (training) alignment",
        "",
        "| Channel | Move change | Reference alignment | Flips toward | Flips away | Net delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("joint", "joint"), ("policy", "policy"), ("value", "value")):
        entry = summary["probe_metrics"][key]
        lines.append(
            f"| {label} | {entry['move_change_rate']:.4f} | {entry['reference_alignment']:.4f} "
            f"| {entry['flips_toward_reference']:.4f} | {entry['flips_away_from_reference']:.4f} "
            f"| {entry['net_reference_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validation (held-out) alignment",
            "",
            "| Channel | Move change | Reference alignment | Flips toward | Flips away | Net delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in (("joint", "joint"), ("policy", "policy"), ("value", "value")):
        entry = summary["validation_metrics"][key]
        lines.append(
            f"| {label} | {entry['move_change_rate']:.4f} | {entry['reference_alignment']:.4f} "
            f"| {entry['flips_toward_reference']:.4f} | {entry['flips_away_from_reference']:.4f} "
            f"| {entry['net_reference_delta']:+.4f} |"
        )
    lines.extend(
        [
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
            "Full evidence: `docs/data/alphazero-lite-search-aware-distillation-channel-summary.json`.",
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
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_search_aware_distillation_channel"),
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-search-aware-distillation-channel-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-search-aware-distillation-channel-results.md",
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
    directions = channel_directions(
        diagnostic_batches, state, optimizer_state, device, manifest
    )

    args.workdir.mkdir(parents=True, exist_ok=True)
    source_metadata = args.current / "metadata.json"
    variants: dict[str, Path] = {
        "current": args.current,
        "joint": write_artifact(
            args.workdir / "artifacts" / "joint",
            directions["joint_state"],
            source_metadata,
        ),
        "policy": write_artifact(
            args.workdir / "artifacts" / "policy",
            directions["policy_state"],
            source_metadata,
        ),
        "value": write_artifact(
            args.workdir / "artifacts" / "value",
            directions["value_state"],
            source_metadata,
        ),
    }

    probe_states = [
        {"state": row["state"], "state_hash": row["state_hash"]} for row in probe_rows
    ]
    validation_states = [
        {"state": row["state"], "state_hash": row["state_hash"]}
        for row in validation_rows
    ]

    probe_moves = search_moves(
        variants, probe_states, sims=MOVE_BUDGET, workers=args.workers
    )
    validation_moves = search_moves(
        variants, validation_states, sims=MOVE_BUDGET, workers=args.workers
    )
    probe_reference = search_moves(
        {"current": variants["current"]},
        probe_states,
        sims=REFERENCE_BUDGET,
        workers=args.workers,
    )
    validation_reference = search_moves(
        {"current": variants["current"]},
        validation_states,
        sims=REFERENCE_BUDGET,
        workers=args.workers,
    )
    probe_ref_map = {
        row["state_hash"]: probe_reference[("current", row["state_hash"])]
        for row in probe_states
    }
    validation_ref_map = {
        row["state_hash"]: validation_reference[("current", row["state_hash"])]
        for row in validation_states
    }

    probe_metrics = {
        name: reference_alignment(probe_moves, name, probe_ref_map, probe_states)
        for name in ("joint", "policy", "value")
    }
    validation_metrics = {
        name: reference_alignment(
            validation_moves, name, validation_ref_map, validation_states
        )
        for name in ("joint", "policy", "value")
    }

    summary: dict[str, Any] = {
        "schema": "azlite_search_aware_distillation_channel_v1",
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
            "reference_budget": REFERENCE_BUDGET,
        },
        "gradient_norms": {
            "policy_grad_norm": directions["policy_grad_norm"],
            "value_grad_norm": directions["value_grad_norm"],
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
