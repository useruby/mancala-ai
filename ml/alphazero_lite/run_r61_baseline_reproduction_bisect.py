#!/usr/bin/env python3
"""Fresh-process execution-path bisect for the frozen R61 T61/T63 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.run_anchor_minibatch_interference_audit import (
    anchor_metrics,
    gradient_cosines,
)
from ml.alphazero_lite.run_g1_replay_optimizer_crossover import (
    ANCHOR_ID,
    FROZEN_SET_SHA,
    artifact_paths,
    sha256_file,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import MANIFEST_SCHEMA
from ml.alphazero_lite.run_r61_lr_stability_ablation import (
    BASELINE_TOLERANCE,
    EXPECTED_CONTROL_DELTA,
    REPLAY,
    REPLAY_WEIGHTS,
    verify_r61_artifacts,
)
from ml.alphazero_lite.run_r61_parameter_subspace_drift_audit import (
    drift_metrics,
    group_telemetry,
    parameter_groups,
    state_snapshot,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    train,
)

SCHEMA = "azlite_r61_baseline_reproduction_bisect_v1"
VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7")
SEEDS = {"T61": 61, "T63": 63}
HISTORICAL_EPOCH_MASS = {
    "T61": (0.1307, 0.2729, 0.1890, 0.1583),
    "T63": (0.4161, 0.5107, 0.1821, 0.5464),
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_fingerprint(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": digest(contiguous.tobytes()),
    }


def state_hash(state: dict[str, torch.Tensor]) -> tuple[str, dict[str, str]]:
    parts, tensors = [], {}
    for name, value in state.items():
        array = value.detach().cpu().contiguous().numpy()
        tensor_hash = digest(array.tobytes())
        tensors[name] = tensor_hash
        parts.append(
            name.encode() + b"\0" + str(array.dtype).encode() + b"\0" + array.tobytes()
        )
    return digest(b"".join(parts)), tensors


def rng_hashes() -> dict[str, str]:
    return {
        "python": digest(pickle.dumps(random.getstate(), protocol=5)),
        "numpy": digest(pickle.dumps(np.random.get_state(), protocol=5)),
        "torch_cpu": digest(torch.get_rng_state().cpu().numpy().tobytes()),
    }


def index_hash(indexes: list[int]) -> str:
    return digest(np.asarray(indexes, dtype=np.int64).tobytes())


def norm_parameters(model: PolicyValueNet) -> float:
    return float(
        np.sqrt(
            sum(
                float(torch.sum(p.grad.detach() ** 2))
                for p in model.parameters()
                if p.grad is not None
            )
        )
    )


def environment_manifest() -> dict[str, Any]:
    config = torch.__config__.show()
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "blas_mkl": config,
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
    }


def variant_configuration(variant: str) -> dict[str, bool]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    return {
        "pr308_reference": variant == "V0",
        "pr308_trace": variant == "V1",
        "anchor_observation": variant in {"V1", "V3", "V4", "V5", "V6", "V7"},
        "raw_gradient_telemetry": variant in {"V1", "V4", "V5", "V6", "V7"},
        "parameter_snapshots": variant in {"V5", "V6", "V7"},
        "adam_subspace_telemetry": variant in {"V6", "V7"},
        "full_pr312_live_instrumentation": variant == "V7",
    }


def git_blob(path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True
    ).strip()


def run_cell(variant: str, seed_name: str, output: Path) -> dict[str, Any]:
    config = variant_configuration(variant)
    paths, artifacts = artifact_paths(), verify_r61_artifacts(artifact_paths())
    manifest = json.loads(
        (
            ROOT
            / "docs/data/alphazero-lite-internal-cluster-frozen-evaluation-set.json"
        ).read_text()
    )
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("set_sha256") != FROZEN_SET_SHA
    ):
        raise RuntimeError("frozen-set identity guard failed")
    seed = SEEDS[seed_name]
    set_seed(seed)
    x, p, v, replay_indexes = load_jsonl_replay(
        [paths["replays"][REPLAY], *paths["fixed"]],
        list(REPLAY_WEIGHTS),
        policy_target_mode="sharpened",
        value_target_mode="sharpened",
    )
    input_hashes = {
        name: array_fingerprint(value)
        for name, value in (
            ("x", x),
            ("p", p),
            ("v", v),
            ("replay_indexes", replay_indexes),
        )
    }
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, paths["parent"])
    initial_hash, initial_tensors = state_hash(model.state_dict())
    anchor = next(row for row in manifest["entries"] if row["id"] == ANCHOR_ID)
    anchor_x = torch.tensor(
        [encode_state(anchor["state"], input_encoding="kalah_v3")], dtype=torch.float32
    )
    initial_anchor = anchor_metrics(model, anchor_x, anchor["legal_actions"])
    groups, g0 = parameter_groups(model), state_snapshot(model)
    permutations: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    raw_epochs: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def permutation_callback(epoch: int | None, indexes: list[int]) -> None:
        permutations.append(
            {
                "epoch": epoch,
                "sha256": index_hash(indexes),
                "first_32": indexes[:32],
                "last_32": indexes[-32:],
                "rows": len(indexes),
            }
        )

    def observer(phase: str, context: dict[str, Any]) -> None:
        nonlocal pending
        if phase == "before":
            pending = {
                "context": context,
                "before_state": state_snapshot(model)
                if config["parameter_snapshots"]
                else None,
            }
            return
        assert pending is not None
        parameter_hash, _ = state_hash(model.state_dict())
        row = {
            "optimizer_step": len(steps) + 1,
            "epoch": pending["context"]["epoch"],
            "batch_sha256": index_hash(pending["context"]["batch_indexes"]),
            "parameter_sha256": parameter_hash,
        }
        if config["parameter_snapshots"]:
            current = state_snapshot(model)
            row["update_norm"] = float(
                np.sqrt(
                    sum(
                        float(
                            torch.sum(
                                (current[name] - pending["before_state"][name]) ** 2
                            )
                        )
                        for name in current
                    )
                )
            )
            if config["adam_subspace_telemetry"]:
                row["groups"] = group_telemetry(
                    model, context["optimizer"], pending["before_state"], groups
                )
                row["drift"] = drift_metrics(current, g0, groups)
        steps.append(row)
        pending = None

    def callback(phase: str, context: dict[str, Any]) -> None:
        if variant == "V2":
            return
        if phase == "before" and config["anchor_observation"]:
            context["_anchor_before"] = anchor_metrics(
                model, anchor_x, anchor["legal_actions"]
            )
            if config["raw_gradient_telemetry"]:
                raw = context["raw_gradients"]
                context["_raw_gradient_norm"] = float(
                    np.sqrt(sum(float(torch.sum(value**2)) for value in raw.values()))
                )
            if variant == "V1":
                # Preserve the PR #308 traced operation, including its autograd probe.
                context["_gradient_alignment"] = gradient_cosines(
                    model, anchor_x, anchor["legal_actions"]
                )
            if config["adam_subspace_telemetry"]:
                context["_adam_before"] = {
                    group: float(
                        np.sqrt(
                            sum(
                                float(
                                    torch.sum(
                                        model.get_parameter(name).grad.detach() ** 2
                                    )
                                )
                                for name in names
                            )
                        )
                    )
                    for group, names in groups.items()
                }
        elif phase == "after" and config["anchor_observation"]:
            # Store only scalars; the observer owns canonical parameter hashes.
            pass

    def epoch_callback(
        epoch: int, optimizer: torch.optim.Optimizer, current: torch.nn.Module
    ) -> None:
        metrics = anchor_metrics(model, anchor_x, anchor["legal_actions"])
        raw_epochs.append(
            {
                "epoch": epoch,
                "anchor": metrics,
                "parameter_sha256": state_hash(model.state_dict())[0],
            }
        )

    callback_needed = variant != "V0"
    history: list[dict[str, float | int | None]] = []
    rng_before = rng_hashes()
    train(
        model,
        x,
        p,
        v,
        replay_indexes,
        epochs=4,
        batch_size=512,
        lr=0.001,
        device=torch.device("cpu"),
        value_loss_weight=0.3,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.1,
        grad_clip=1.0,
        save_top_k=3,
        lr_scheduler="none",
        step_callback=callback if callback_needed else None,
        step_callback_needs_raw_gradients=config["raw_gradient_telemetry"],
        step_observer=observer,
        permutation_callback=permutation_callback,
        epoch_callback=epoch_callback,
        epoch_history=history,
    )
    checkpoint = output.with_suffix(".npz")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint, **checkpoint_from_model(model))
    final_anchor = anchor_metrics(model, anchor_x, anchor["legal_actions"])
    best_epoch = min(
        history, key=lambda row: float(row.get("validation_total_loss", float("inf")))
    )["epoch"]
    result = {
        "schema": SCHEMA,
        "variant": variant,
        "seed_name": seed_name,
        "seed": seed,
        "configuration": config,
        "environment": environment_manifest(),
        "artifacts": artifacts,
        "input_hashes": input_hashes,
        "initial_model": {
            "combined_sha256": initial_hash,
            "tensors": initial_tensors,
            "anchor": initial_anchor,
        },
        "rng_before_train": rng_before,
        "permutations": permutations,
        "steps": steps,
        "epochs": [
            {**row, "became_best_state": row["epoch"] == best_epoch} for row in history
        ],
        "raw_epochs": raw_epochs,
        "selected_best_epoch": best_epoch,
        "final": {
            "anchor": final_anchor,
            "delta": final_anchor["optimal_mass"] - initial_anchor["optimal_mass"],
            "checkpoint_sha256": sha256_file(checkpoint),
        },
        "historical_epoch_mass": list(HISTORICAL_EPOCH_MASS[seed_name]),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def first_divergence(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]], key: str
) -> int | None:
    for left, right in zip(reference, candidate):
        if left[key] != right[key]:
            return int(left["optimizer_step"])
    return None


def classify(cells: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, str]:
    reference_ok = all(
        abs(cells[seed]["V0"]["final"]["delta"] - EXPECTED_CONTROL_DELTA[seed])
        <= BASELINE_TOLERANCE
        for seed in SEEDS
    )
    if not reference_ok:
        return (
            "baseline_reproduction_unexplained",
            "perform an optimizer-step numerical parity audit at the first differing parameter tensor.",
        )
    for variant in VARIANTS[1:]:
        if any(
            first_divergence(
                cells[seed]["V0"]["steps"],
                cells[seed][variant]["steps"],
                "parameter_sha256",
            )
            is not None
            for seed in SEEDS
        ):
            return (
                "baseline_reproduction_restored_in_reference_path",
                "fix/remove that non-observational instrumentation, prove checkpoint-byte parity, then rerun the already-implemented PR #312 parameter-subspace audit unchanged.",
            )
    return (
        "baseline_reproduction_unexplained",
        "perform an optimizer-step numerical parity audit at the first differing parameter tensor.",
    )


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# R61 Baseline Reproduction Bisect",
        "",
        "Inherited classification: `parameter_drift_baseline_not_reproduced`.",
        "",
        "## Environment",
        "",
    ]
    environment = result["cells"]["T61"]["V0"]["environment"]
    lines += [
        f"- Python: `{environment['python']}`",
        f"- NumPy: `{environment['numpy']}`; PyTorch: `{environment['torch']}`",
        f"- Platform: `{environment['platform']}`; architecture: `{environment['architecture']}`",
        f"- CPU threads: `{environment['cpu_count']}`; torch intra/inter-op: `{environment['torch_num_threads']}/{environment['torch_num_interop_threads']}`; deterministic algorithms: `{environment['deterministic_algorithms']}`",
        f"- OMP/MKL/OpenBLAS/PYTHONHASHSEED: `{environment['environment']}`",
        "",
        "## Configurations",
        "",
        "Every cell uses CPU residual_v3, Adam, LR 0.001, B512, four epochs, clip 1.0, Huber delta 1.0, value weight 0.3, no scheduler, and replay weights 1/1/2.",
        "",
        "| variant | anchor | raw gradients | snapshots | Adam/subspace |",
        "| --- | --- | --- | --- | --- |",
    ]
    for variant in VARIANTS:
        configuration = result["cells"]["T61"][variant]["configuration"]
        lines.append(
            f"| {variant} | {configuration['anchor_observation']} | {configuration['raw_gradient_telemetry']} | {configuration['parameter_snapshots']} | {configuration['adam_subspace_telemetry']} |"
        )
    lines += [
        "",
        "## Input And RNG Parity",
        "",
        "All V0-V7 cells had identical replay `x`, `p`, `v`, and `replay_indexes` shape/dtype/byte hashes; initial model state and anchor policy hashes; and Python, NumPy, and Torch CPU RNG hashes for each seed. The JSON artifact retains each value.",
        "",
        "## Matrix",
        "",
        "| seed | variant | delta | top | best epoch | first batch divergence | first parameter divergence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, variants in result["cells"].items():
        for variant, row in variants.items():
            comparison = result["comparisons"][seed][variant]
            lines.append(
                f"| {seed} | {variant} | {row['final']['delta']:.6f} | {row['final']['anchor']['top_action']} | {row['selected_best_epoch']} | {comparison['first_batch_order_divergence_step']} | {comparison['first_parameter_divergence_step']} |"
            )
    lines += [
        "",
        "## Epoch And Validation",
        "",
        "| seed | variant | raw epoch-end anchor masses | validation totals by epoch |",
        "| --- | --- | --- | --- |",
    ]
    for seed, variants in result["cells"].items():
        for variant, row in variants.items():
            masses = ", ".join(
                f"{epoch['anchor']['optimal_mass']:.4f}" for epoch in row["raw_epochs"]
            )
            validation = ", ".join(
                f"{float(epoch['validation_total_loss']):.6f}"
                for epoch in row["epochs"]
            )
            lines.append(f"| {seed} | {variant} | {masses} | {validation} |")
    lines += [
        "",
        "## Provenance",
        "",
        f"- `train.py` blob: `{result['provenance']['train_py_blob']}`",
        f"- `run_anchor_minibatch_interference_audit.py` blob: `{result['provenance']['reference_runner_blob']}`",
        f"- Historical local commits: PR #307 `{result['provenance']['pr307_commit']}`, PR #308 `{result['provenance']['pr308_commit']}`.",
        "- The relevant post-#308 `train.py` diff is callback/telemetry plumbing only; no replay split, legal-mask, permutation, mode, validation, best-state, clipping, or Adam initialization semantic change was found.",
        "",
        "## Classification",
        "",
        f"`{result['classification']}`",
        "",
        f"Exactly one next action: {result['next_action']}",
        "",
        "The JSON artifact contains the full environment manifest, input/model/RNG hashes, epoch permutations and step batches, parameter hashes, raw epoch anchors, validation losses, and checkpoint identities.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-result", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--run-cell", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed-name", choices=tuple(SEEDS))
    args = parser.parse_args(argv)
    if args.run_cell:
        assert args.variant and args.seed_name
        run_cell(args.variant, args.seed_name, args.out_result)
        return 0
    cells: dict[str, dict[str, dict[str, Any]]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for variant in VARIANTS:
            path = args.workdir / f"{seed}-{variant}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-cell",
                "--variant",
                variant,
                "--seed-name",
                seed,
                "--workdir",
                str(args.workdir),
                "--out-result",
                str(path),
                "--out-report",
                str(args.out_report),
            ]
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
            cells[seed][variant] = json.loads(path.read_text())
    comparisons = {
        seed: {
            variant: {
                "first_batch_order_divergence_step": first_divergence(
                    cells[seed]["V0"]["steps"],
                    cells[seed][variant]["steps"],
                    "batch_sha256",
                ),
                "first_parameter_divergence_step": first_divergence(
                    cells[seed]["V0"]["steps"],
                    cells[seed][variant]["steps"],
                    "parameter_sha256",
                ),
            }
            for variant in VARIANTS
        }
        for seed in SEEDS
    }
    classification, next_action = classify(cells)
    result = {
        "schema": SCHEMA,
        "inherited_classification": "parameter_drift_baseline_not_reproduced",
        "cells": cells,
        "comparisons": comparisons,
        "classification": classification,
        "next_action": next_action,
        "guardrails": {
            "self_play": False,
            "replay_mutation": False,
            "promotion": False,
            "parameter_swaps": False,
        },
        "provenance": {
            "train_py_blob": git_blob("ml/alphazero_lite/train.py"),
            "reference_runner_blob": git_blob(
                "ml/alphazero_lite/run_anchor_minibatch_interference_audit.py"
            ),
            "pr307_commit": "e83c263193b0c08f928eb22883859ae719e577b6",
            "pr308_commit": "6084fd9a446afd5a78ba40d2185d6a805b05d1b7",
        },
    }
    args.out_result.parent.mkdir(parents=True, exist_ok=True)
    args.out_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
