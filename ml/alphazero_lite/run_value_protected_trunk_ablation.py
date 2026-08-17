#!/usr/bin/env python3
"""Run the prespecified one-sided value-protected shared-trunk ablation.

The baseline uses the unmodified PR191 backward path.  The protected lane only
replaces conflicting policy gradients on input_layer/residual_layers, before
the existing global norm clip and Adam step.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
    write_fixed_npz,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    ARENA_SUITE,
    CURRENT_HASH,
    JOINT_HASH,
    _suite_provenance,
    _batch,
    _losses,
    _new_model,
    _save_snapshot,
    export_snapshot_artifacts,
    output_drift,
    puct_trajectory,
)
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    parse_game_jsonl,
    run_arena,
)
from ml.alphazero_lite.run_shared_trunk_delta_attribution import _context_c_puct  # noqa: E402
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    decoded_validation_manifest,
    stable_hash,
)
from ml.alphazero_lite.run_terminal_outcome_selfplay_iteration_smoke import (  # noqa: E402
    export_checkpoint,
)
from ml.alphazero_lite.train import checkpoint_from_model, load_checkpoint_into_model  # noqa: E402

TRUNK_PREFIXES = ("input_layer.", "residual_layers.")
DENSE_STEPS = (0, 1, 2, 3, 4, 5, 8, 12)
FULL_STEPS = (23, 35, 46)
EPSILON = 1e-20


def is_trunk_parameter(name: str) -> bool:
    """Return whether a parameter is part of the shared residual trunk."""
    return name.startswith(TRUNK_PREFIXES)


def parameter_family(name: str) -> str:
    """Return the only family permitted to receive each gradient component."""
    if is_trunk_parameter(name):
        return "trunk"
    if name.startswith("policy_"):
        return "policy_head"
    if name.startswith("value_"):
        return "value_head"
    raise ValueError(f"unexpected residual_v3 parameter: {name}")


def _flat(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.detach().reshape(-1) for value in values])


def project_policy_gradient(
    policy: list[torch.Tensor], value: list[torch.Tensor], *, enabled: bool
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    """Project the global trunk policy vector only when it opposes value."""
    policy_vector, value_vector = _flat(policy), _flat(value)
    dot = torch.dot(policy_vector, value_vector)
    policy_norm = torch.linalg.vector_norm(policy_vector)
    value_norm = torch.linalg.vector_norm(value_vector)
    fired = bool(enabled and dot < 0)
    scale = (
        dot / (torch.dot(value_vector, value_vector) + EPSILON) if fired else dot * 0
    )
    projected = [
        gradient - scale * value_gradient
        for gradient, value_gradient in zip(policy, value, strict=True)
    ]
    projected_vector = _flat(projected)
    removed = _flat(
        [left - right for left, right in zip(policy, projected, strict=True)]
    )
    return projected, {
        "raw_policy_value_dot": float(dot),
        "raw_policy_value_cosine": float(dot / (policy_norm * value_norm + EPSILON)),
        "projection_fired": fired,
        "projection_scale": float(scale),
        "removed_policy_gradient_norm": float(torch.linalg.vector_norm(removed)),
        "removed_policy_gradient_fraction": float(
            torch.linalg.vector_norm(removed) / (policy_norm + EPSILON)
        ),
        "post_projection_policy_value_cosine": float(
            torch.dot(projected_vector, value_vector)
            / (torch.linalg.vector_norm(projected_vector) * value_norm + EPSILON)
        ),
        "combined_trunk_gradient_norm": float(
            torch.linalg.vector_norm(projected_vector + value_vector)
        ),
    }


def trunk_group(name: str) -> str:
    """Map shared parameters to the requested layer/block telemetry group."""
    if name.startswith("input_layer."):
        return "input_layer"
    if name.startswith("residual_layers."):
        return f"residual_block_{name.split('.')[1]}"
    raise ValueError(f"not a trunk parameter: {name}")


def trunk_group_telemetry(
    named: list[tuple[str, torch.nn.Parameter]],
    policy: list[torch.Tensor],
    value: list[torch.Tensor],
    *,
    scale: float,
) -> dict[str, dict[str, float]]:
    """Report raw and post-projection conflict separately for every trunk group."""
    groups: dict[str, list[int]] = {}
    for index, (name, _parameter) in enumerate(named):
        if is_trunk_parameter(name):
            groups.setdefault(trunk_group(name), []).append(index)
    result = {}
    for group, indexes in groups.items():
        policy_vector = _flat([policy[index] for index in indexes])
        value_vector = _flat([value[index] for index in indexes])
        projected = policy_vector - scale * value_vector
        result[group] = {
            "raw_policy_value_cosine": float(
                torch.dot(policy_vector, value_vector)
                / (
                    torch.linalg.vector_norm(policy_vector)
                    * torch.linalg.vector_norm(value_vector)
                    + EPSILON
                )
            ),
            "post_projection_policy_value_cosine": float(
                torch.dot(projected, value_vector)
                / (
                    torch.linalg.vector_norm(projected)
                    * torch.linalg.vector_norm(value_vector)
                    + EPSILON
                )
            ),
        }
    return result


def _loss_gradients(
    model: torch.nn.Module, batch: dict[str, torch.Tensor]
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[tuple[str, torch.nn.Parameter]],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    policy_loss, value_loss = _losses(model, batch)
    named = list(model.named_parameters())
    parameters = [parameter for _name, parameter in named]
    policy = torch.autograd.grad(
        policy_loss, parameters, retain_graph=True, allow_unused=True
    )
    value = torch.autograd.grad(
        value_loss, parameters, retain_graph=True, allow_unused=True
    )
    zeros = [torch.zeros_like(parameter) for _name, parameter in named]
    return (
        policy_loss,
        value_loss,
        named,
        [
            gradient.detach() if gradient is not None else zero
            for gradient, zero in zip(policy, zeros, strict=True)
        ],
        [
            gradient.detach() if gradient is not None else zero
            for gradient, zero in zip(value, zeros, strict=True)
        ],
    )


def compose_gradients(
    named: list[tuple[str, torch.nn.Parameter]],
    policy: list[torch.Tensor],
    value: list[torch.Tensor],
    *,
    projection_enabled: bool,
) -> dict[str, float | bool]:
    """Assign component gradients, preserving both private-head gradients exactly."""
    trunk_indexes = [
        index
        for index, (name, _parameter) in enumerate(named)
        if is_trunk_parameter(name)
    ]
    projected, telemetry = project_policy_gradient(
        [policy[index] for index in trunk_indexes],
        [value[index] for index in trunk_indexes],
        enabled=projection_enabled,
    )
    for index, (name, parameter) in enumerate(named):
        family = parameter_family(name)
        if family == "trunk":
            parameter.grad = projected[trunk_indexes.index(index)] + value[index]
        elif family == "policy_head":
            parameter.grad = policy[index].clone()
        else:
            parameter.grad = value[index].clone()
    return telemetry


def _global_grad_norm(model: torch.nn.Module) -> float:
    return float(
        torch.linalg.vector_norm(
            _flat(
                [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
            )
        )
    )


def optimizer_virtual_updates(
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    lr: float,
    clip: float,
    projection_enabled: bool,
) -> dict[str, float]:
    """Measure independent Adam U_policy, U_value, and U_joint at one boundary."""
    updates: dict[str, torch.Tensor] = {}
    for kind in ("policy", "value", "joint"):
        model = _new_model(device)
        model.load_state_dict(state)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if is_trunk_parameter(name)
        }
        optimizer.zero_grad(set_to_none=True)
        policy_loss, value_loss, named, policy, value = _loss_gradients(model, batch)
        if kind == "joint" and projection_enabled:
            compose_gradients(named, policy, value, projection_enabled=True)
        elif kind == "policy":
            policy_loss.backward()
        elif kind == "value":
            value_loss.backward()
        else:
            (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        updates[kind] = _flat(
            [
                (parameter.detach() - before[name]).reshape(-1)
                for name, parameter in model.named_parameters()
                if is_trunk_parameter(name)
            ]
        )
    return {
        f"adam_u_{kind}_norm": float(torch.linalg.vector_norm(vector))
        for kind, vector in updates.items()
    } | {
        "adam_u_policy_value_cosine": float(
            torch.dot(updates["policy"], updates["value"])
            / (
                torch.linalg.vector_norm(updates["policy"])
                * torch.linalg.vector_norm(updates["value"])
                + EPSILON
            )
        ),
        "adam_u_joint_policy_cosine": float(
            torch.dot(updates["joint"], updates["policy"])
            / (
                torch.linalg.vector_norm(updates["joint"])
                * torch.linalg.vector_norm(updates["policy"])
                + EPSILON
            )
        ),
        "adam_u_joint_value_cosine": float(
            torch.dot(updates["joint"], updates["value"])
            / (
                torch.linalg.vector_norm(updates["joint"])
                * torch.linalg.vector_norm(updates["value"])
                + EPSILON
            )
        ),
    }


def optimizer_telemetry(
    snapshots: dict[
        int,
        tuple[dict[str, torch.Tensor], dict[str, Any]],
    ],
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    lr: float,
    clip: float,
    projection_enabled: bool,
) -> dict[str, Any]:
    """Repeat isolated virtual-step measurements on the PR193 fixed batch set."""
    result = {}
    for step, (state, optimizer_state) in snapshots.items():
        samples = [
            optimizer_virtual_updates(
                state,
                optimizer_state,
                batch,
                device,
                lr=lr,
                clip=clip,
                projection_enabled=projection_enabled,
            )
            for batch in batches[:32]
        ]
        result[str(step)] = {
            key: float(np.mean([sample[key] for sample in samples]))
            for key in samples[0]
        }
    return result


def _weights_hash(state: dict[str, torch.Tensor], workdir: Path, label: str) -> str:
    model = _new_model(torch.device("cpu"))
    model.load_state_dict(state)
    checkpoint = workdir / f"{label}.npz"
    write_fixed_npz(checkpoint, checkpoint_from_model(model))
    artifact = workdir / f"{label}_artifact"
    export_checkpoint(
        checkpoint_path=checkpoint,
        out_dir=artifact,
        version=label,
        policy_loss=0.0,
        value_loss=0.0,
    )
    return sha256_file(artifact / "weights.json")


def replay_lane(
    manifest: dict[str, Any],
    workdir: Path,
    device: torch.device,
    *,
    projection_enabled: bool,
    stop_at: int,
) -> tuple[
    dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Replay exactly the saved PR191 batches, optionally projecting trunk policy gradients."""
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(paths["train_source_indexes"], allow_pickle=False)
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    batches = [
        _batch([rows[int(index)] for index in source], indexes, device)
        for indexes in plan
    ]
    model = _new_model(device)
    load_checkpoint_into_model(model, paths["initialization_checkpoint"])
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(manifest["optimizer"]["lr"]),
        weight_decay=float(manifest["optimizer"].get("weight_decay", 0.0)),
    )
    wanted = {step for step in (*DENSE_STEPS, *FULL_STEPS) if step <= stop_at}
    snapshots = {
        0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)
    }
    telemetry: list[dict[str, Any]] = []
    for step, batch in enumerate(batches[:stop_at], 1):
        optimizer.zero_grad(set_to_none=True)
        policy_loss, value_loss, named, policy, value = _loss_gradients(model, batch)
        if projection_enabled:
            record: dict[str, Any] = compose_gradients(
                named, policy, value, projection_enabled=True
            )
        else:
            # Retain the PR191 backward order and arithmetic for the hash reproduction.
            (policy_loss + value_loss).backward()
            trunk = [
                index
                for index, (name, _parameter) in enumerate(named)
                if is_trunk_parameter(name)
            ]
            _projected, record = project_policy_gradient(
                [policy[index] for index in trunk],
                [value[index] for index in trunk],
                enabled=False,
            )
        record["step"] = step
        record["by_trunk_group"] = trunk_group_telemetry(
            named, policy, value, scale=float(record["projection_scale"])
        )
        record["combined_trunk_gradient_norm"] = (
            _global_grad_norm(model)
            if not projection_enabled
            else record["combined_trunk_gradient_norm"]
        )
        pre_clip = _global_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(manifest["gradient_clip"])
        )
        record["post_clip_norm"] = _global_grad_norm(model)
        record["pre_clip_global_norm"] = pre_clip
        telemetry.append(record)
        optimizer.step()
        if step in wanted:
            snapshots[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )
    return snapshots, batches, telemetry


def aggregate_telemetry(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate the preregistered real-batch telemetry ranges."""
    ranges = {"1-4": (1, 4), "5-12": (5, 12), "13-23": (13, 23), "24-46": (24, 46)}
    result = {}
    for label, (first, last) in ranges.items():
        rows = [row for row in samples if first <= int(row["step"]) <= last]
        if rows:
            aggregate: dict[str, Any] = {
                key: float(np.mean([float(row[key]) for row in rows]))
                for key in rows[0]
                if key not in {"step", "by_trunk_group"}
            }
            aggregate["by_trunk_group"] = {
                group: {
                    key: float(
                        np.mean([row["by_trunk_group"][group][key] for row in rows])
                    )
                    for key in rows[0]["by_trunk_group"][group]
                }
                for group in rows[0]["by_trunk_group"]
            }
            result[label] = aggregate
    return result


def _lane_evidence(
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    rows: list[dict[str, Any]],
    workdir: Path,
    probe: list[dict[str, Any]],
    manifest_hash: str,
    *,
    puct: bool,
) -> dict[str, Any]:
    result = {"output": output_drift(rows, snapshots)}
    if puct:
        artifacts = export_snapshot_artifacts(snapshots, workdir)
        # PR193's frozen search probe is the first fixed 256 validated states.
        result["puct"] = puct_trajectory(probe[:256], artifacts, workdir, manifest_hash)
    return result


def same_step_difference(
    protected: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Compute the prespecified protected-minus-baseline trajectory contrast."""
    result: dict[str, Any] = {"output": {}}
    for step in protected["output"]:
        if step not in baseline["output"]:
            continue
        result["output"][step] = {
            family: {
                metric: protected["output"][step][family][metric]
                - baseline["output"][step][family][metric]
                for metric in protected["output"][step][family]
            }
            for family in ("policy", "value")
        }
    if "puct" in protected and "puct" in baseline:
        result["puct"] = {
            step: {
                context: {
                    metric: protected["puct"]["metrics"][step][context][metric]
                    - baseline["puct"]["metrics"][step][context][metric]
                    for metric in protected["puct"]["metrics"][step][context]
                    if metric != "states"
                }
                for context in protected["puct"]["metrics"][step]
            }
            for step in protected["puct"]["metrics"]
            if step in baseline["puct"]["metrics"]
        }
    return result


def _arena_records(
    workdir: Path,
    challenger: Path,
    incumbent: Path,
    context: str,
    role: str,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    """Run/cached paired-seat canonical games for exactly one model comparison."""
    sims, current_sims = (int(value) for value in context.split(":"))
    result = []
    for seat in (0, 1):
        directory = workdir / context.replace(":", "_") / role / f"starts_{seat}"
        records = directory / "arena.jsonl"
        if not records.is_file() or len(parse_game_jsonl(str(records))) != 256:
            directory.mkdir(parents=True, exist_ok=True)
            run_arena(
                challenger=str(challenger),
                current=str(incumbent),
                challenger_sims=sims,
                current_sims=current_sims,
                games=256,
                seed=42,
                workers=workers,
                out_json=str(directory / "arena.json"),
                out_jsonl=str(records),
                opening_prefixes_jsonl=str(ARENA_SUITE),
                challenger_starts=seat,
                games_per_opening=2,
                root_policy_mode="deterministic",
                root_temperature=0.0,
                normalize_values=False,
                c_puct=_context_c_puct(context),
                tactical_root_bias=0.0,
                seed_ledger_output=str(directory / "seed_ledger.jsonl"),
            )
        result.extend(parse_game_jsonl(str(records)))
    return result


def early_arena(
    artifacts: dict[str, dict[int, Path]], current: Path, workdir: Path, *, workers: int
) -> dict[str, Any]:
    """Evaluate both lanes and direct protected-minus-baseline effects per opening."""
    suite = _suite_provenance(ARENA_SUITE)
    metrics: dict[str, Any] = {}
    for step in (1, 3, 5, 12):
        for context in ("384:256", "1200:1200"):
            current_control = _arena_records(
                workdir / "current_control",
                current,
                current,
                context,
                "current_control",
                workers=workers,
            )
            baseline_current = _arena_records(
                workdir / f"step_{step:04d}",
                artifacts["baseline_joint"][step],
                current,
                context,
                "baseline_vs_current",
                workers=workers,
            )
            protected_current = _arena_records(
                workdir / f"step_{step:04d}",
                artifacts["value_protected_joint"][step],
                current,
                context,
                "protected_vs_current",
                workers=workers,
            )
            protected_baseline = _arena_records(
                workdir / f"step_{step:04d}",
                artifacts["value_protected_joint"][step],
                artifacts["baseline_joint"][step],
                context,
                "protected_vs_baseline",
                workers=workers,
            )
            metrics.setdefault(str(step), {})[context] = {
                "baseline_minus_current": paired_opening_candidate_effect(
                    baseline_current,
                    current_control,
                    bootstrap_samples=10_000,
                    bootstrap_seed=42,
                ),
                "protected_minus_current": paired_opening_candidate_effect(
                    protected_current,
                    current_control,
                    bootstrap_samples=10_000,
                    bootstrap_seed=42,
                ),
                "protected_minus_baseline": paired_opening_candidate_effect(
                    protected_baseline,
                    baseline_current,
                    bootstrap_samples=10_000,
                    bootstrap_seed=42,
                ),
            }
    return {"suite": suite, "metrics": metrics}


def markdown(summary: dict[str, Any]) -> str:
    """Render the completed or gated protocol record."""
    phase = summary["phase_h"]
    lines = [
        "# AlphaZero-Lite Value-Protected Trunk Ablation",
        "",
        f"**Classification:** `{summary['classification']}`",
        "",
        f"- baseline PR191 hash reproduced: `{summary['phase_a']['reproduced_exactly']}`",
        f"- protected steps completed: `{summary['protected_steps_completed']}`",
        f"- projection firing rate through step 12: `{phase['projection_firing_rate']:.4f}`",
        f"- continuation gate: `{phase['passed']}`",
        f"- continuation decision: `{phase['reason']}`",
        "",
    ]
    arena = summary.get("phase_g_early_arena", {}).get("metrics", {})
    if "12" in arena:
        lines.extend(
            [
                "## Step-12 Arena Gate",
                "",
                "| Context | Protected - baseline | Protected - current |",
                "| --- | ---: | ---: |",
            ]
        )
        for context, metrics in arena["12"].items():
            direct = metrics["protected_minus_baseline"]["paired_candidate_effect"]
            current = metrics["protected_minus_current"]["paired_candidate_effect"]
            lines.append(f"| {context} | {direct:+.4f} | {current:+.4f} |")
        lines.append("")
    lines.extend(
        [
            "Full replay snapshots (including optimizer states), real-batch conflict telemetry, frozen-probe metrics, and paired arena evidence are in `docs/data/alphazero-lite-value-protected-trunk-ablation-summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_value_protected_trunk_ablation"),
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
        / "docs/data/alphazero-lite-value-protected-trunk-ablation-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-value-protected-trunk-ablation-results.md",
    )
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--optimizer-audit", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument("--full", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact does not match the PR191 initialization")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, int(manifest["seed"]))
    baseline, batches, baseline_telemetry = replay_lane(
        manifest,
        args.workdir / "baseline_joint",
        device,
        projection_enabled=False,
        stop_at=len(
            np.load(
                Path(manifest["artifact_paths"]["batch_indexes"]), allow_pickle=False
            )
        ),
    )
    baseline_hash = _weights_hash(
        baseline[max(baseline)][0], args.workdir, "baseline_joint_final"
    )
    if baseline_hash != JOINT_HASH:
        raise RuntimeError(
            f"PR191 replay hash mismatch: expected {JOINT_HASH}, got {baseline_hash}"
        )
    configure_determinism(device, int(manifest["seed"]))
    protected, _batches, protected_telemetry = replay_lane(
        manifest,
        args.workdir / "value_protected_joint",
        device,
        projection_enabled=True,
        stop_at=12,
    )
    rows = read_jsonl(Path(manifest["replay_path"]))
    validation = np.load(
        Path(manifest["artifact_paths"]["validation_source_indexes"]),
        allow_pickle=False,
    )
    probe, probe_manifest = decoded_validation_manifest(rows, validation)
    probe_manifest["manifest_sha256"] = stable_hash(probe_manifest)
    evidence = {
        "baseline_joint": _lane_evidence(
            baseline,
            rows,
            args.workdir / "baseline_joint",
            probe,
            probe_manifest["manifest_sha256"],
            puct=args.puct,
        ),
        "value_protected_joint": _lane_evidence(
            protected,
            rows,
            args.workdir / "value_protected_joint",
            probe,
            probe_manifest["manifest_sha256"],
            puct=args.puct,
        ),
    }
    evidence["protected_minus_baseline"] = same_step_difference(
        evidence["value_protected_joint"], evidence["baseline_joint"]
    )
    adam = {}
    if args.optimizer_audit:
        audit_steps = {step: baseline[step] for step in DENSE_STEPS}
        protected_audit_steps = {step: protected[step] for step in DENSE_STEPS}
        adam = {
            "baseline_joint": optimizer_telemetry(
                audit_steps,
                batches,
                device,
                lr=float(manifest["optimizer"]["lr"]),
                clip=float(manifest["gradient_clip"]),
                projection_enabled=False,
            ),
            "value_protected_joint": optimizer_telemetry(
                protected_audit_steps,
                batches,
                device,
                lr=float(manifest["optimizer"]["lr"]),
                clip=float(manifest["gradient_clip"]),
                projection_enabled=True,
            ),
        }
    firing_rate = float(
        np.mean([row["projection_fired"] for row in protected_telemetry])
    )
    arena = {}
    phase_h = {
        "projection_firing_rate": firing_rate,
        "mechanistically_inactive": firing_rate < 0.05,
        "passed": False,
    }
    if args.arena:
        artifacts = {
            "baseline_joint": export_snapshot_artifacts(
                baseline, args.workdir / "baseline_joint"
            ),
            "value_protected_joint": export_snapshot_artifacts(
                protected, args.workdir / "value_protected_joint"
            ),
        }
        arena = early_arena(
            artifacts,
            args.current,
            args.workdir / "early_arena",
            workers=args.arena_workers,
        )
        step12 = arena["metrics"]["12"]
        protected_better = all(
            item["protected_minus_baseline"]["paired_candidate_effect"] > 0
            for item in step12.values()
        )
        protected_safe = all(
            item["protected_minus_current"]["opening_bootstrap_ci"]["upper_95"] >= -0.03
            for item in step12.values()
        )
        phase_h["passed"] = bool(
            firing_rate >= 0.05 and protected_better and protected_safe
        )
        phase_h["protected_better_at_both_step12_contexts"] = protected_better
        phase_h["protected_vs_current_upper_ci_safe"] = protected_safe
        phase_h["reason"] = (
            "passed" if phase_h["passed"] else "prespecified_step_12_arena_gate_failed"
        )
    else:
        phase_h["reason"] = "arena_gate_not_run"
    summary: dict[str, Any] = {
        "schema": "azlite_value_protected_trunk_ablation_v1",
        "inputs": {
            "current_weights_sha256": CURRENT_HASH,
            "expected_joint_weights_sha256": JOINT_HASH,
            "training_manifest": str(args.pr191_workdir / "training_manifest.json"),
        },
        "guardrails": {
            "value_weight": 0.6,
            "gradient_clip": 1.0,
            "projection": "one_sided_policy_on_shared_trunk_only",
            "no_random_task_ordering": True,
        },
        "phase_a": {
            "final_weights_sha256": baseline_hash,
            "reproduced_exactly": True,
            "batch_count": len(batches),
        },
        "snapshot_steps": {
            "baseline_joint": sorted(baseline),
            "value_protected_joint": sorted(protected),
        },
        "telemetry": {
            "baseline_joint": {
                "batches": baseline_telemetry,
                "ranges": aggregate_telemetry(baseline_telemetry),
            },
            "value_protected_joint": {
                "batches": protected_telemetry,
                "ranges": aggregate_telemetry(protected_telemetry),
            },
        },
        "probe_manifest": probe_manifest,
        "optimizer_aware_telemetry": adam,
        "evidence": evidence,
        "phase_g_early_arena": arena,
        "phase_h": phase_h,
        "protected_steps_completed": 12,
        "classification": "mechanistically_inactive"
        if firing_rate < 0.05
        else "continuation_gate_failed"
        if args.arena and not phase_h["passed"]
        else "early_protocol_complete_awaiting_arena_gate",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline_hash": baseline_hash,
                "classification": summary["classification"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
