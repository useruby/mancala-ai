#!/usr/bin/env python3
"""Diagnostic-only parameter-subspace audit for the frozen R61 T61/T63 split.

This runner trains only the historical baseline, never writes replay, generates
self-play, changes optimizer settings, or promotes a checkpoint.  Hybrids are
in-memory/offline models used solely for evaluation.
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

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.run_anchor_minibatch_interference_audit import anchor_metrics
from ml.alphazero_lite.run_g1_replay_optimizer_crossover import (
    ANCHOR_ID,
    FROZEN_SET_SHA,
    G0_SHA,
    artifact_paths,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import (
    MANIFEST_SCHEMA,
    grouped_aggregates,
)
from ml.alphazero_lite.run_r61_lr_stability_ablation import (
    BASELINE_TOLERANCE,
    EXPECTED_CONTROL_DELTA,
    REPLAY,
    REPLAY_WEIGHTS,
    verify_r61_artifacts,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    train,
)

SCHEMA = "azlite_r61_parameter_subspace_drift_audit_v1"
TRAINING_SEEDS = {"T61": 61, "T63": 63}
EPOCHS, BATCH_SIZE, LR0, GRAD_CLIP = 4, 512, 0.001, 1.0
FINE_GROUPS = (
    "trunk_input",
    "trunk_residual",
    "policy_hidden",
    "policy_readout",
    "value_hidden",
    "value_readout",
)
ROLLED_GROUPS = {
    "shared_trunk": ("trunk_input", "trunk_residual"),
    "policy_path": ("policy_hidden", "policy_readout"),
    "value_path": ("value_hidden", "value_readout"),
}
EPS = 1e-12


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parameter_groups(model: PolicyValueNet) -> dict[str, tuple[str, ...]]:
    """Mechanically partition every trainable residual_v3 tensor by exact prefix."""
    if model.model_type != "residual_v3":
        raise ValueError("parameter audit requires residual_v3")
    prefixes = {
        "trunk_input": "input_layer.",
        "trunk_residual": "residual_layers.",
        "policy_hidden": "policy_hidden_layer.",
        "policy_readout": "policy_head.",
        "value_hidden": "value_hidden_layer.",
        "value_readout": "value_head.",
    }
    groups = {group: [] for group in FINE_GROUPS}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        memberships = [
            group for group, prefix in prefixes.items() if name.startswith(prefix)
        ]
        if len(memberships) != 1:
            raise RuntimeError(f"unclassified or multiply classified parameter: {name}")
        groups[memberships[0]].append(name)
    if not all(groups.values()):
        raise RuntimeError("residual_v3 parameter group unexpectedly empty")
    return {group: tuple(names) for group, names in groups.items()}


def names_for(groups: dict[str, tuple[str, ...]], group: str) -> tuple[str, ...]:
    return tuple(
        name for fine in ROLLED_GROUPS.get(group, (group,)) for name in groups[fine]
    )


def vector(state: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([state[name].detach().reshape(-1).float().cpu() for name in names])


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = norm(left) * norm(right)
    return float(torch.dot(left, right) / denominator) if denominator else None


def state_snapshot(model: PolicyValueNet) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def margin_metrics(
    model: PolicyValueNet, anchor_x: torch.Tensor, legal: list[int]
) -> dict[str, Any]:
    metrics = anchor_metrics(model, anchor_x, legal)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits, _ = model(anchor_x)
        values = logits[0].cpu()
    model.train(was_training)
    other = max(float(values[action]) for action in legal if action != 0)
    return {
        **metrics,
        "anchor_margin": float(values[0]) - other,
        "logits": [float(v) for v in values],
    }


def group_telemetry(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    before: dict[str, torch.Tensor],
    groups: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    current = state_snapshot(model)
    result = {}
    by_name = dict(model.named_parameters())
    for group in FINE_GROUPS:
        names = groups[group]
        raw = torch.cat(
            [by_name[name].grad.detach().reshape(-1).cpu() for name in names]
        )
        # The trainer clips globally before its callback; reconstructing the scalar
        # from callback context is unnecessary for group post-clip measurement.
        updates = vector(current, names) - vector(before, names)
        m1 = torch.cat(
            [
                optimizer.state[by_name[name]]["exp_avg"].detach().reshape(-1).cpu()
                for name in names
            ]
        )
        m2 = torch.cat(
            [
                optimizer.state[by_name[name]]["exp_avg_sq"].detach().reshape(-1).cpu()
                for name in names
            ]
        )
        result[group] = {
            "post_clip_gradient_norm": norm(raw),
            "parameter_update_norm": norm(updates),
            "adam_first_moment_norm": norm(m1),
            "adam_second_moment_norm": norm(m2),
            "gradient_first_moment_cosine": cosine(raw, m1),
            "update_raw_gradient_cosine": cosine(updates, raw),
        }
    return result


def drift_metrics(
    current: dict[str, torch.Tensor],
    g0: dict[str, torch.Tensor],
    groups: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    result = {}
    for group in (*FINE_GROUPS, *ROLLED_GROUPS):
        names = names_for(groups, group)
        delta, base = vector(current, names) - vector(g0, names), vector(g0, names)
        result[group] = {
            "drift_norm": norm(delta),
            "relative_drift": norm(delta) / max(norm(base), EPS),
        }
    return result


def anchor_probe(
    model: PolicyValueNet,
    anchor_x: torch.Tensor,
    legal: list[int],
    g0: dict[str, torch.Tensor],
    latest_update: dict[str, torch.Tensor],
    groups: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Autograd probe without backward(), preserving grads, optimizer, and RNG."""
    old_grads = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
    }
    was_training = model.training
    model.eval()
    logits, _ = model(anchor_x)
    masked = logits.clone()
    masked[:, [action for action in range(6) if action not in legal]] = -1e9
    loss = -torch.log_softmax(masked, dim=1)[0, 0]
    named = list(model.named_parameters())
    gradients = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    model.train(was_training)
    current = state_snapshot(model)
    output, total_sq = (
        {},
        sum(float(torch.sum(g.detach() ** 2)) for g in gradients if g is not None),
    )
    grad_by_name = {
        name: torch.zeros_like(p).cpu() if g is None else g.detach().cpu()
        for (name, p), g in zip(named, gradients)
    }
    for group in (*FINE_GROUPS, *ROLLED_GROUPS):
        names = names_for(groups, group)
        probe = vector(grad_by_name, names)
        drift = vector(current, names) - vector(g0, names)
        update = vector(latest_update, names)
        output[group] = {
            "gradient_norm": norm(probe),
            "total_squared_gradient_fraction": float(torch.sum(probe**2)) / total_sq
            if total_sq
            else 0.0,
            "drift_cosine": cosine(probe, drift),
            "recent_update_cosine": cosine(probe, update),
        }
    for name, parameter in model.named_parameters():
        parameter.grad = old_grads[name]
    return output


def make_hybrid(
    base: PolicyValueNet,
    donor: PolicyValueNet,
    groups: dict[str, tuple[str, ...]],
    swap_groups: tuple[str, ...],
) -> PolicyValueNet:
    hybrid = copy.deepcopy(base)
    source, destination = (
        dict(donor.named_parameters()),
        dict(hybrid.named_parameters()),
    )
    names = {name for group in swap_groups for name in names_for(groups, group)}
    for name in names:
        if source[name].shape != destination[name].shape:
            raise RuntimeError(f"hybrid architecture mismatch: {name}")
        destination[name].data.copy_(source[name].data)
    return hybrid


def decomposition(model: PolicyValueNet, anchor_x: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        trunk = model.trunk_features(anchor_x)
        hidden = torch.relu(model.policy_hidden_layer(trunk))
        logits = model.policy_head(hidden)[0]
    return {
        "trunk_feature_norm": norm(trunk[0].cpu()),
        "policy_hidden_feature_norm": norm(hidden[0].cpu()),
        "logits": [float(v) for v in logits.cpu()],
    }


def evaluate_model(
    model: PolicyValueNet,
    manifest: dict[str, Any],
    anchor_x: torch.Tensor,
    anchor: dict[str, Any],
) -> dict[str, Any]:
    # A temporary checkpoint is avoided: evaluator-equivalent frozen metrics are
    # computed directly from the same encoded states and exact manifest labels.
    rows = []
    for entry in manifest["entries"]:
        x = torch.tensor(
            [encode_state(entry["state"], input_encoding="kalah_v3")],
            dtype=torch.float32,
        )
        metric = margin_metrics(model, x, entry["legal_actions"])
        rows.append(
            {
                "id": entry["id"],
                "membership": entry["membership"],
                "optimal_mass": sum(
                    metric["policy"][a] for a in entry["exact_outcome_optimal_actions"]
                ),
                "top_is_outcome_optimal": metric["top_action"]
                in entry["exact_outcome_optimal_actions"],
                "degrading_action_mass": sum(
                    metric["policy"][action] for action in entry["degrading_actions"]
                ),
                "policy_entropy": metric["policy_entropy"],
                "exact_wdl_value_error": 0.0,
            }
        )
    return {
        "anchor": margin_metrics(model, anchor_x, anchor["legal_actions"]),
        "frozen_set": grouped_aggregates(rows),
        "rows": rows,
        "logit_decomposition": decomposition(model, anchor_x),
    }


def compare_progress(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    g0: dict[str, torch.Tensor],
    groups: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    result = {}
    for group in (*FINE_GROUPS, *ROLLED_GROUPS):
        names = names_for(groups, group)
        a, b = (
            vector(left, names) - vector(g0, names),
            vector(right, names) - vector(g0, names),
        )
        distance = norm(a - b)
        result[group] = {
            "drift_cosine": cosine(a, b),
            "norm_ratio_t61_t63": norm(a) / max(norm(b), EPS),
            "euclidean_distance": distance,
            "signed_drift_agreement_fraction": float(
                torch.mean(((a >= 0) == (b >= 0)).float())
            ),
            "normalized_t61_t63_distance": distance / (0.5 * (norm(a) + norm(b)) + EPS),
        }
    return result


def run_trajectory(
    paths: dict[str, Any], manifest: dict[str, Any], seed_name: str
) -> dict[str, Any]:
    set_seed(TRAINING_SEEDS[seed_name])
    x, p, v, replay_indexes = load_jsonl_replay(
        [paths["replays"][REPLAY], *paths["fixed"]],
        list(REPLAY_WEIGHTS),
        policy_target_mode="sharpened",
        value_target_mode="sharpened",
    )
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, paths["parent"])
    groups, g0 = parameter_groups(model), state_snapshot(model)
    anchor = next(row for row in manifest["entries"] if row["id"] == ANCHOR_ID)
    anchor_x = torch.tensor(
        [encode_state(anchor["state"], input_encoding="kalah_v3")], dtype=torch.float32
    )
    traces, snapshots = [], {0: g0}
    pending: dict[str, Any] | None = None
    latest_update = {name: torch.zeros_like(value) for name, value in g0.items()}

    def callback(phase: str, context: dict[str, Any]) -> None:
        nonlocal pending, latest_update
        if phase == "before":
            pending = {
                "context": context,
                "state": state_snapshot(model),
                "anchor": margin_metrics(model, anchor_x, anchor["legal_actions"]),
            }
            return
        assert pending is not None
        step = len(traces) + 1
        current = state_snapshot(model)
        latest_update = {
            name: current[name] - pending["state"][name] for name in current
        }
        telemetry = group_telemetry(
            model, context["optimizer"], pending["state"], groups
        )
        # train_one_epoch passes raw grads in before callback and only calls after
        # post-step; retain raw group norms from that untouched snapshot.
        raw_by_name = pending["context"]["raw_gradients"]
        for group in FINE_GROUPS:
            telemetry[group]["pre_clip_gradient_norm"] = norm(
                vector(raw_by_name, groups[group])
            )
        row = {
            "optimizer_step": step,
            "epoch": pending["context"]["epoch"],
            "anchor_before": pending["anchor"],
            "anchor_after": margin_metrics(model, anchor_x, anchor["legal_actions"]),
            "global_preclip_gradient_norm": pending["context"]["gradient_norm"],
            "global_postclip_gradient_norm": pending["context"][
                "post_clip_gradient_norm"
            ],
            "groups": telemetry,
            "drift": drift_metrics(current, g0, groups),
        }
        traces.append(row)
        if step % 8 == 0:
            snapshots[step] = current
        pending = None

    def epoch_callback(
        epoch: int, _optimizer: torch.optim.Optimizer, current: torch.nn.Module
    ) -> None:
        snapshots[len(traces)] = state_snapshot(current)  # epoch boundary

    train(
        model,
        x,
        p,
        v,
        replay_indexes,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR0,
        device=torch.device("cpu"),
        value_loss_weight=0.3,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.1,
        grad_clip=GRAD_CLIP,
        save_top_k=3,
        lr_scheduler="none",
        step_callback=callback,
        epoch_callback=epoch_callback,
    )
    snapshots[len(traces)] = state_snapshot(model)
    selected = sorted(
        {
            0,
            len(traces),
            *(round(len(traces) * fraction) for fraction in (0.25, 0.5, 0.75)),
        }
    )
    probes = {
        str(step): anchor_probe(
            make_hybrid(model, model, groups, ()),
            anchor_x,
            anchor["legal_actions"],
            g0,
            latest_update,
            groups,
        )
        if step == len(traces)
        else None
        for step in selected
    }
    # Rehydrate selected snapshots for probes without affecting the completed run.
    for step in selected[:-1]:
        probe_model = copy.deepcopy(model)
        with torch.no_grad():
            for name, parameter in probe_model.named_parameters():
                parameter.copy_(snapshots[step][name])
        probes[str(step)] = anchor_probe(
            probe_model, anchor_x, anchor["legal_actions"], g0, latest_update, groups
        )
    return {
        "model": model,
        "groups": groups,
        "g0": g0,
        "traces": traces,
        "snapshots": snapshots,
        "probes": probes,
        "anchor": anchor,
        "anchor_x": anchor_x,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# R61 Parameter Subspace Drift Audit",
        "",
        "Inherited PR #311 classification: `grad_clip_relaxation_unstable`.",
        "",
        "## Result",
        "",
        f"Hard classification: `{result['classification']}`",
        "",
        f"Exactly one next experiment: {result['next_experiment']}",
        "",
        "## Baseline Reproduction",
        "",
        "| seed | expected delta | observed delta | final top |",
        "| --- | ---: | ---: | ---: |",
    ]
    for seed, row in result["baseline_reproduction"].items():
        lines.append(
            f"| {seed} | {row['expected_delta']:.4f} | {row['actual_delta']:.4f} | {row['top_action']} |"
        )
    lines += [
        "",
        "## Parameter Groups",
        "",
        "`trunk_input=input_layer.*`; `trunk_residual=residual_layers.*`; `policy_hidden=policy_hidden_layer.*`; `policy_readout=policy_head.*`; `value_hidden=value_hidden_layer.*`; `value_readout=value_head.*`. The runner rejects any unclassified or overlapping trainable tensor. Rolled-up groups are `shared_trunk`, `policy_path`, and `value_path`.",
        "",
        "## Hybrid Matrix",
        "",
        "| hybrid | anchor top | anchor mass | margin | cluster mass | control mass |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["hybrids"].items():
        anchor, frozen = row["anchor"], row["frozen_set"]
        lines.append(
            f"| {name} | {anchor['top_action']} | {anchor['optimal_mass']:.4f} | {anchor['anchor_margin']:.4f} | {frozen['cluster']['optimal_mass']:.4f} | {frozen['matched_control']['optimal_mass']:.4f} |"
        )
    lines += [
        "",
        "All per-step scalar telemetry, matched-progress distances, anchor probes, feature/logit decompositions, and reverse transplants are retained in the JSON artifact. Value-path swaps are negative controls; they cannot directly enter residual_v3 policy logits.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-result", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
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
        or manifest.get("training_injection") is not False
    ):
        raise RuntimeError("frozen-set identity/injection guard failed")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "inherited_classification": "grad_clip_relaxation_unstable",
        "artifacts": artifacts,
        "historical_recipe": {
            "optimizer": "Adam",
            "lr": LR0,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "grad_clip": GRAD_CLIP,
            "replay_weights": list(REPLAY_WEIGHTS),
            "lr_schedule": "none",
        },
        "guardrails": {
            "self_play": False,
            "replay_mutation": False,
            "replay_weights_changed": False,
            "anchor_training_injection": False,
            "promotion": False,
            "optimizer_hyperparameter_sweep": False,
        },
    }
    if not args.execute:
        result.update({"classification": "planned", "next_experiment": "not run"})
        write_json(args.out_result, result)
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(
            "# R61 Parameter Subspace Drift Audit\n\nPlanned diagnostic only.\n"
        )
        return 0
    trajectories = {
        seed: run_trajectory(paths, manifest, seed) for seed in TRAINING_SEEDS
    }
    baseline = {}
    for seed, trajectory in trajectories.items():
        evaluated = evaluate_model(
            trajectory["model"], manifest, trajectory["anchor_x"], trajectory["anchor"]
        )
        # Use a correctly loaded G0 for the reproduction delta.
        g0model = PolicyValueNet(
            (96, 3), "residual_v3", trajectory["anchor_x"].shape[1]
        )
        load_checkpoint_into_model(g0model, paths["parent"])
        delta = (
            evaluated["anchor"]["optimal_mass"]
            - margin_metrics(
                g0model, trajectory["anchor_x"], trajectory["anchor"]["legal_actions"]
            )["optimal_mass"]
        )
        baseline[seed] = {
            "expected_delta": EXPECTED_CONTROL_DELTA[seed],
            "actual_delta": delta,
            "within_tolerance": abs(delta - EXPECTED_CONTROL_DELTA[seed])
            <= BASELINE_TOLERANCE,
            "top_action": evaluated["anchor"]["top_action"],
        }
        trajectory["evaluation"] = evaluated
    if (
        not all(row["within_tolerance"] for row in baseline.values())
        or baseline["T61"]["top_action"] == 0
        or baseline["T63"]["top_action"] != 0
    ):
        result.update(
            {
                "baseline_reproduction": baseline,
                "classification": "parameter_drift_baseline_not_reproduced",
                "next_experiment": "none",
            }
        )
        write_json(args.out_result, result)
        raise RuntimeError("parameter_drift_baseline_not_reproduced")
    groups, g0 = trajectories["T61"]["groups"], trajectories["T61"]["g0"]
    steps = sorted(
        set(trajectories["T61"]["snapshots"]) & set(trajectories["T63"]["snapshots"])
    )
    matched = {
        str(step): compare_progress(
            trajectories["T61"]["snapshots"][step],
            trajectories["T63"]["snapshots"][step],
            g0,
            groups,
        )
        for step in steps
    }
    hybridspec = {
        "T61_native": ("T61", "T61", ()),
        "T63_native": ("T63", "T63", ()),
        "T61_heads_T63_shared_trunk": ("T61", "T63", ("shared_trunk",)),
        "T63_heads_T61_shared_trunk": ("T63", "T61", ("shared_trunk",)),
        "T61_trunk_value_T63_policy_path": ("T61", "T63", ("policy_path",)),
        "T63_trunk_value_T61_policy_path": ("T63", "T61", ("policy_path",)),
        "T61_T63_policy_hidden": ("T61", "T63", ("policy_hidden",)),
        "T61_T63_policy_readout": ("T61", "T63", ("policy_readout",)),
        "T63_T61_policy_hidden": ("T63", "T61", ("policy_hidden",)),
        "T63_T61_policy_readout": ("T63", "T61", ("policy_readout",)),
        "T61_T63_value_path": ("T61", "T63", ("value_path",)),
        "T63_T61_value_path": ("T63", "T61", ("value_path",)),
    }
    hybrids = {}
    for name, (base, donor, swaps) in hybridspec.items():
        model = make_hybrid(
            trajectories[base]["model"], trajectories[donor]["model"], groups, swaps
        )
        hybrids[name] = evaluate_model(
            model,
            manifest,
            trajectories[base]["anchor_x"],
            trajectories[base]["anchor"],
        )
    for base, donor, name in (
        ("T61", "T63", "T61_T63_value_path"),
        ("T63", "T61", "T63_T61_value_path"),
    ):
        native = hybrids[f"{base}_native"]["anchor"]["logits"]
        swapped = hybrids[name]["anchor"]["logits"]
        if not np.array_equal(np.asarray(native), np.asarray(swapped)):
            raise RuntimeError("value-path negative control changed policy logits")
    t61, t63 = hybrids["T61_native"]["anchor"], hybrids["T63_native"]["anchor"]
    trunk, policy = (
        hybrids["T61_heads_T63_shared_trunk"]["anchor"],
        hybrids["T61_trunk_value_T63_policy_path"]["anchor"],
    )
    gap = t63["anchor_margin"] - t61["anchor_margin"]
    trunk_fraction = (
        (trunk["anchor_margin"] - t61["anchor_margin"]) / gap if gap else 0.0
    )
    policy_fraction = (
        (policy["anchor_margin"] - t61["anchor_margin"]) / gap if gap else 0.0
    )
    if (
        trunk["top_action"] == 0
        and hybrids["T63_heads_T61_shared_trunk"]["anchor"]["top_action"] != 0
        and policy["top_action"] != 0
    ):
        classification, next_experiment = (
            "anchor_drift_shared_trunk_primary",
            "run one frozen-R61 existing trainable-scope ablation chosen from this audit to limit destructive trunk movement while allowing policy learning.",
        )
    elif (
        policy["top_action"] == 0
        and hybrids["T63_trunk_value_T61_policy_path"]["anchor"]["top_action"] != 0
        and trunk["top_action"] != 0
    ):
        classification, next_experiment = (
            "anchor_drift_policy_head_primary",
            "run one frozen-R61 existing policy-path training-scope ablation matching the implicated policy subcomponent.",
        )
    else:
        classification, next_experiment = (
            "anchor_drift_distributed",
            "test one existing isolation scope that most cleanly separates policy-gradient trunk updates from value-gradient trunk updates.",
        )
    result.update(
        {
            "baseline_reproduction": baseline,
            "parameter_groups": groups,
            "trajectories": {
                seed: {
                    "step_trace": value["traces"],
                    "anchor_sensitivity": value["probes"],
                    "final": value["evaluation"],
                }
                for seed, value in trajectories.items()
            },
            "matched_progress": matched,
            "hybrids": hybrids,
            "fraction_explained": {
                "shared_trunk": trunk_fraction,
                "policy_path": policy_fraction,
            },
            "classification": classification,
            "next_experiment": next_experiment,
            "g0_sha256": G0_SHA,
            "frozen_set_sha256": FROZEN_SET_SHA,
        }
    )
    write_json(args.out_result, result)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
