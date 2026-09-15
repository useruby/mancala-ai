#!/usr/bin/env python3
"""Diagnostic-only policy/value gradient attribution for the frozen R61 split.

The runner performs the original full-scope T61/T63 runs twice: a reference
run and a live instrumented run.  The observer is pre-backward and never
alters production gradients, optimizer state, data ordering, or replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.run_g1_replay_optimizer_crossover import (
    ANCHOR_ID,
    FROZEN_SET_SHA,
    G0_SHA,
    artifact_paths,
    sha256_file,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import MANIFEST_SCHEMA
from ml.alphazero_lite.run_r61_lr_stability_ablation import (
    REPLAY,
    REPLAY_WEIGHTS,
    verify_r61_artifacts,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    checkpoint_from_model,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    train,
)

SCHEMA = "azlite_r61_policy_value_trunk_gradient_audit_v1"
TRAINING_SEEDS = {"T61": 61, "T63": 63}
EXPECTED_SHA = {
    "T61": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "T63": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
}
EPOCHS, BATCH_SIZE, LR, GRAD_CLIP, VALUE_WEIGHT = 4, 512, 0.001, 1.0, 0.3
FINE_GROUPS = (
    "trunk_input",
    "residual_block_0",
    "residual_block_1",
    "residual_block_2",
    "policy_hidden",
    "policy_readout",
    "value_hidden",
    "value_readout",
)
SHARED_TRUNK = FINE_GROUPS[:4]
EPS = 1e-12
IDENTITY_TOLERANCE = 2e-6


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parameter_groups(model: PolicyValueNet) -> dict[str, tuple[str, ...]]:
    """Partition residual_v3 tensors, retaining each residual block separately."""
    prefixes = {
        "trunk_input": "input_layer.",
        "residual_block_0": "residual_layers.0.",
        "residual_block_1": "residual_layers.1.",
        "residual_block_2": "residual_layers.2.",
        "policy_hidden": "policy_hidden_layer.",
        "policy_readout": "policy_head.",
        "value_hidden": "value_hidden_layer.",
        "value_readout": "value_head.",
    }
    groups = {name: [] for name in FINE_GROUPS}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            matches = [
                group for group, prefix in prefixes.items() if name.startswith(prefix)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"unclassified or overlapping trainable parameter: {name}"
                )
            groups[matches[0]].append(name)
    if not all(groups.values()):
        raise RuntimeError("unexpected empty R61 parameter group")
    return {name: tuple(values) for name, values in groups.items()}


def group_names(groups: dict[str, tuple[str, ...]], group: str) -> tuple[str, ...]:
    return tuple(
        name
        for item in (SHARED_TRUNK if group == "shared_trunk" else (group,))
        for name in groups[item]
    )


def snapshot(model: PolicyValueNet) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def vector(values: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
    return torch.cat(
        [values[name].detach().reshape(-1).float().cpu() for name in names]
    )


def scalar_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = scalar_norm(left) * scalar_norm(right)
    return None if denominator == 0 else float(torch.dot(left, right) / denominator)


def percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(values, percent)) if values else 0.0


def legal_margin(
    model: PolicyValueNet, x: torch.Tensor, legal: list[int]
) -> torch.Tensor:
    logits, _ = model(x)
    return (
        logits[0, 0]
        - torch.stack([logits[0, action] for action in legal if action != 0]).max()
    )


def probe_loss(model: PolicyValueNet, rows: list[dict[str, Any]]) -> torch.Tensor:
    """Evaluation-only exact-label probe; it is never included in training loss."""
    losses = []
    for row in rows:
        x = torch.tensor(
            [encode_state(row["state"], input_encoding="kalah_v3")], dtype=torch.float32
        )
        logits, _ = model(x)
        legal = row["legal_actions"]
        masked = logits[0, legal]
        probabilities = torch.softmax(masked, dim=0)
        optimal = torch.tensor(
            [action in row["exact_outcome_optimal_actions"] for action in legal]
        )
        losses.append(-torch.log(probabilities[optimal].sum()))
    return torch.stack(losses).mean()


def gradients(loss: torch.Tensor, model: PolicyValueNet) -> dict[str, torch.Tensor]:
    named = list(model.named_parameters())
    values = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named],
        retain_graph=True,
        allow_unused=True,
    )
    return {
        name: (torch.zeros_like(parameter) if value is None else value.detach())
        for (name, parameter), value in zip(named, values)
    }


def component_metrics(
    policy: dict[str, torch.Tensor],
    value: dict[str, torch.Tensor],
    total: dict[str, torch.Tensor],
    margin_gradient: dict[str, torch.Tensor],
    cluster_gradient: dict[str, torch.Tensor],
    groups: dict[str, tuple[str, ...]],
    clip_scale: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in (*FINE_GROUPS, "shared_trunk"):
        names = group_names(groups, group)
        p, v, t, margin, cluster = (
            vector(item, names)
            for item in (policy, value, total, margin_gradient, cluster_gradient)
        )
        pn, vn, tn = scalar_norm(p), scalar_norm(v), scalar_norm(t)
        # A negative parameter step has positive margin effect when it opposes dmargin/dtheta.
        result[group] = {
            "policy_preclip_norm": pn,
            "value_weighted_preclip_norm": vn,
            "total_preclip_norm": tn,
            "policy_postclip_norm": clip_scale * pn,
            "value_weighted_postclip_norm": clip_scale * vn,
            "total_postclip_norm": clip_scale * tn,
            "policy_squared_component_fraction": float(
                torch.dot(p, p) / max(torch.dot(t, t), torch.tensor(EPS))
            ),
            "value_squared_component_fraction": float(
                torch.dot(v, v) / max(torch.dot(t, t), torch.tensor(EPS))
            ),
            "policy_value_norm_ratio": pn / max(vn, EPS),
            "policy_value_cosine": cosine(p, v),
            "policy_anchor_effect": float(torch.dot(-clip_scale * p, margin)),
            "value_anchor_effect": float(torch.dot(-clip_scale * v, margin)),
            "policy_cluster_effect": float(torch.dot(-clip_scale * p, cluster)),
            "value_cluster_effect": float(torch.dot(-clip_scale * v, cluster)),
        }
    return result


def aggregate(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    values = [row["groups"][group] for row in rows]

    def numbers(key: str) -> list[float]:
        return [float(item[key]) for item in values if item[key] is not None]

    cosines = numbers("policy_value_cosine")
    output = {
        "steps": len(values),
        "mean_cosine": statistics.fmean(cosines) if cosines else 0.0,
        "median_cosine": percentile(cosines, 50),
        "p10_cosine": percentile(cosines, 10),
        "p90_cosine": percentile(cosines, 90),
        "fraction_conflicting": statistics.fmean(value < -0.25 for value in cosines)
        if cosines
        else 0.0,
    }
    for key in (
        "policy_preclip_norm",
        "value_weighted_preclip_norm",
        "policy_anchor_effect",
        "value_anchor_effect",
        "policy_cluster_effect",
        "value_cluster_effect",
    ):
        output[f"mean_{key}"] = statistics.fmean(numbers(key)) if values else 0.0
    return output


def contribution(rows: list[dict[str, Any]], group: str, key: str) -> dict[str, float]:
    policy = [float(row["groups"][group][f"policy_{key}"]) for row in rows]
    value = [float(row["groups"][group][f"value_{key}"]) for row in rows]

    def summary(items: list[float], prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_helpful": sum(item for item in items if item > 0),
            f"{prefix}_harmful": sum(-item for item in items if item < 0),
            f"{prefix}_net": sum(items),
        }

    return summary(policy, "policy") | summary(value, "value")


def checkpoint(model: PolicyValueNet, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **checkpoint_from_model(model))
    return sha256_file(path)


def cloned_adam_step(
    before: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    batch_x: np.ndarray,
    batch_p: np.ndarray,
    batch_v: np.ndarray,
    *,
    source: str,
) -> PolicyValueNet:
    """Run one discarded historical/no-source-trunk Adam step.

    Heads always retain their historical combined gradient. Only shared-trunk
    tensors are replaced for the two local counterfactuals, then the modified
    *full* gradient is globally clipped before Adam steps.
    """
    if source not in {"historical", "no_value_to_trunk", "no_policy_to_trunk"}:
        raise ValueError(f"unsupported counterfactual source: {source}")
    model = PolicyValueNet((96, 3), "residual_v3", batch_x.shape[1])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(before[name])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    x = torch.from_numpy(batch_x)
    target_p, target_v = torch.from_numpy(batch_p), torch.from_numpy(batch_v)
    mask = torch.from_numpy(legal_mask_matrix_for_encoded_states(batch_x))
    logits, prediction = model(x)
    policy_loss = compute_policy_cross_entropy(
        logits.masked_fill(mask <= 0.0, -1e9), target_p
    ).mean()
    value_loss = compute_value_loss_vector(
        prediction, target_v, value_loss="huber", huber_delta=1.0
    ).mean()
    policy, value = (
        gradients(policy_loss, model),
        {
            name: VALUE_WEIGHT * item
            for name, item in gradients(value_loss, model).items()
        },
    )
    for name, parameter in model.named_parameters():
        is_trunk = name.startswith(("input_layer.", "residual_layers."))
        if source == "no_value_to_trunk" and is_trunk:
            parameter.grad = policy[name].clone()
        elif source == "no_policy_to_trunk" and is_trunk:
            parameter.grad = value[name].clone()
        else:
            parameter.grad = (policy[name] + value[name]).clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return model


def counterfactual_summary(
    lane: dict[str, Any],
    compact_x: np.ndarray,
    compact_p: np.ndarray,
    compact_v: np.ndarray,
    manifest: dict[str, Any],
    selected: list[int],
) -> list[dict[str, Any]]:
    """Validate discarded clone A before interpreting one-step B/C probes."""
    anchor = next(row for row in manifest["entries"] if row["id"] == ANCHOR_ID)
    anchor_x = torch.tensor(
        [encode_state(anchor["state"], input_encoding="kalah_v3")], dtype=torch.float32
    )
    cluster = [
        row for row in manifest["entries"] if row["membership"] != "matched_control"
    ]
    output = []
    for index in selected:
        row = lane["traces"][index]
        sample_indexes = np.asarray(row["batch_indexes"], dtype=np.int64)
        common = (
            lane["states"][index],
            lane["optimizer_states"][index],
            compact_x[sample_indexes],
            compact_p[sample_indexes],
            compact_v[sample_indexes],
        )
        variants = {
            source: cloned_adam_step(*common, source=source)
            for source in ("historical", "no_value_to_trunk", "no_policy_to_trunk")
        }
        actual = lane["after_states"][index]
        maximum_error = max(
            float((parameter.detach().cpu() - actual[name]).abs().max())
            for name, parameter in variants["historical"].named_parameters()
        )
        if maximum_error > IDENTITY_TOLERANCE:
            raise RuntimeError("adam_counterfactual_not_validated")
        metrics = {}
        for source, model in variants.items():
            with torch.no_grad():
                metrics[source] = {
                    "anchor_margin": float(
                        legal_margin(model, anchor_x, anchor["legal_actions"])
                    ),
                    "cluster_probe_loss": float(probe_loss(model, cluster)),
                }
        output.append(
            {
                "optimizer_step": row["optimizer_step"],
                "historical_max_abs_tensor_error": maximum_error,
                "variants": metrics,
                "remove_value_effect": metrics["no_value_to_trunk"]["anchor_margin"]
                - metrics["historical"]["anchor_margin"],
                "remove_policy_effect": metrics["no_policy_to_trunk"]["anchor_margin"]
                - metrics["historical"]["anchor_margin"],
                "remove_value_cluster_effect": metrics["historical"][
                    "cluster_probe_loss"
                ]
                - metrics["no_value_to_trunk"]["cluster_probe_loss"],
                "remove_policy_cluster_effect": metrics["historical"][
                    "cluster_probe_loss"
                ]
                - metrics["no_policy_to_trunk"]["cluster_probe_loss"],
            }
        )
    return output


def run_lane(
    paths: dict[str, Any],
    manifest: dict[str, Any],
    seed_name: str,
    workdir: Path,
    instrumented: bool,
) -> dict[str, Any]:
    set_seed(TRAINING_SEEDS[seed_name])
    x, p, v, indexes = load_jsonl_replay(
        [paths["replays"][REPLAY], *paths["fixed"]],
        list(REPLAY_WEIGHTS),
        policy_target_mode="sharpened",
        value_target_mode="sharpened",
    )
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, paths["parent"])
    groups = parameter_groups(model)
    anchor = next(row for row in manifest["entries"] if row["id"] == ANCHOR_ID)
    anchor_x = torch.tensor(
        [encode_state(anchor["state"], input_encoding="kalah_v3")], dtype=torch.float32
    )
    cluster = [
        row for row in manifest["entries"] if row["membership"] != "matched_control"
    ]
    control = [
        row for row in manifest["entries"] if row["membership"] == "matched_control"
    ]
    traces: list[dict[str, Any]] = []
    states: list[dict[str, torch.Tensor]] = []
    after_states: list[dict[str, torch.Tensor]] = []
    optimizer_states: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}
    permutations: dict[str, str] = {}
    history: list[dict[str, float | int | None]] = []

    def permutation(epoch: int | None, values: list[int]) -> None:
        permutations[str(epoch)] = hashlib.sha256(
            json.dumps(values).encode()
        ).hexdigest()

    def loss_observer(context: dict[str, Any]) -> None:
        if not instrumented:
            return
        if (
            context["pairwise_loss_weight"] != 0.0
            or context["behavior_loss_weight"] != 0.0
        ):
            raise RuntimeError(
                "loss_gradient_attribution_missing_nonzero_loss_component"
            )
        policy = gradients(context["policy_loss_tensor"], model)
        raw_value = gradients(context["value_loss_tensor"], model)
        value = {name: VALUE_WEIGHT * item for name, item in raw_value.items()}
        total = gradients(context["total_loss_tensor"], model)
        was_training = model.training
        model.eval()
        margin_gradient = gradients(
            legal_margin(model, anchor_x, anchor["legal_actions"]), model
        )
        cluster_gradient = gradients(probe_loss(model, cluster), model)
        control_gradient = gradients(probe_loss(model, control), model)
        model.train(was_training)
        pvec, vvec, tvec = (
            vector(item, tuple(name for names in groups.values() for name in names))
            for item in (policy, value, total)
        )
        identity = tvec - pvec - vvec
        clip_scale = min(1.0, GRAD_CLIP / max(scalar_norm(tvec), EPS))
        pending["diagnostic"] = {
            "groups": component_metrics(
                policy,
                value,
                total,
                margin_gradient,
                cluster_gradient,
                groups,
                clip_scale,
            ),
            "control_groups": component_metrics(
                policy,
                value,
                total,
                margin_gradient,
                control_gradient,
                groups,
                clip_scale,
            ),
            "identity_max_abs_error": float(identity.abs().max()),
            "identity_rms_error": float(torch.sqrt(torch.mean(identity.square()))),
            "clip_scale": clip_scale,
            "global_preclip_norm": scalar_norm(tvec),
            "policy": {name: item.cpu().clone() for name, item in policy.items()},
            "value": {name: item.cpu().clone() for name, item in value.items()},
            "margin_gradient": {
                name: item.cpu().clone() for name, item in margin_gradient.items()
            },
        }

    def step_callback(phase: str, context: dict[str, Any]) -> None:
        nonlocal pending
        if phase == "before":
            pending["before"] = snapshot(model)
            pending["optimizer"] = copy.deepcopy(context["optimizer"].state_dict())
            pending["margin_before"] = float(
                legal_margin(model, anchor_x, anchor["legal_actions"]).detach()
            )
            return
        before = pending.pop("before")
        after = snapshot(model)
        row: dict[str, Any] = {
            "optimizer_step": len(traces) + 1,
            "epoch": context["epoch"],
            "batch_indexes": context["batch_indexes"],
            "anchor_margin_before": pending.pop("margin_before"),
            "anchor_margin_after": float(
                legal_margin(model, anchor_x, anchor["legal_actions"]).detach()
            ),
        }
        if instrumented:
            diagnostic = pending.pop("diagnostic")
            if diagnostic["identity_max_abs_error"] > IDENTITY_TOLERANCE:
                raise RuntimeError("loss_gradient_attribution_identity_failed")
            diagnostic["actual_update"] = {}
            for group in (*FINE_GROUPS, "shared_trunk"):
                names = group_names(groups, group)
                update = vector(after, names) - vector(before, names)
                policy, value = (
                    vector(diagnostic["policy"], names),
                    vector(diagnostic["value"], names),
                )
                diagnostic["actual_update"][group] = {
                    "update_norm": scalar_norm(update),
                    "cosine_to_negative_policy": cosine(update, -policy),
                    "cosine_to_negative_value": cosine(update, -value),
                    "cosine_to_negative_total": cosine(update, -(policy + value)),
                    "dot_actual_update_anchor_margin_gradient": float(
                        torch.dot(
                            update,
                            vector(diagnostic["margin_gradient"], names),
                        )
                    ),
                }
            row |= diagnostic
            states.append(before)
            after_states.append(after)
            optimizer_states.append(pending["optimizer"])
        traces.append(row)

    train(
        model,
        x,
        p,
        v,
        indexes,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=torch.device("cpu"),
        value_loss_weight=VALUE_WEIGHT,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.1,
        grad_clip=GRAD_CLIP,
        save_top_k=3,
        lr_scheduler="none",
        step_callback=step_callback,
        step_callback_needs_raw_gradients=False,
        loss_observer=loss_observer if instrumented else None,
        permutation_callback=permutation,
        epoch_history=history,
    )
    path = (
        workdir / ("instrumented" if instrumented else "reference") / f"{seed_name}.npz"
    )
    return {
        "model": model,
        "groups": groups,
        "traces": traces,
        "states": states,
        "after_states": after_states,
        "optimizer_states": optimizer_states,
        "compact_x": x,
        "compact_p": p,
        "compact_v": v,
        "permutations": permutations,
        "epoch_history": history,
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint(model, path),
    }


def classify(
    t61: dict[str, float],
    t63: dict[str, float],
    t61_cluster: dict[str, float],
    t63_cluster: dict[str, float],
    counterfactuals: list[dict[str, Any]],
) -> tuple[str, str]:
    total_harmful = max(t61["policy_harmful"] + t61["value_harmful"], EPS)
    policy_share = t61["policy_harmful"] / total_harmful
    value_share = t61["value_harmful"] / total_harmful
    value_rescue = statistics.fmean(
        row["remove_value_effect"] > 0.0 for row in counterfactuals
    )
    policy_rescue = statistics.fmean(
        row["remove_policy_effect"] > 0.0 for row in counterfactuals
    )
    if (
        value_share >= 0.60
        and t61_cluster["value_harmful"] > t61_cluster["policy_harmful"]
        and t61["value_harmful"] > t63["value_harmful"]
        and t61_cluster["value_harmful"] > t63_cluster["value_harmful"]
        and value_rescue >= 0.70
        and policy_rescue < 0.70
    ):
        return (
            "shared_trunk_value_gradient_primary",
            "run frozen-R61 detach_value_trunk causal training ablation for T61 and T63 only.",
        )
    if (
        policy_share >= 0.60
        and t61_cluster["policy_harmful"] > t61_cluster["value_harmful"]
        and t61["policy_harmful"] > t63["policy_harmful"]
        and t61_cluster["policy_harmful"] > t63_cluster["policy_harmful"]
        and policy_rescue >= 0.70
        and value_rescue < 0.70
    ):
        return (
            "shared_trunk_policy_gradient_primary",
            "run frozen-R61 policy-to-trunk detach causal training ablation for T61 and T63 only.",
        )
    return (
        "shared_trunk_loss_gradient_not_explanatory",
        "move to activation/representation drift on the frozen cluster across the original full T61/T63 trajectories.",
    )


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# R61 Policy vs Value Shared-Trunk Gradient Audit",
        "",
        "Inherited #315: `anchor_drift_shared_trunk_primary`. Inherited #316: `heads_only_no_anchor_rescue`. Inherited #317: `last_block_policy_no_anchor_rescue`.",
        "",
        "## Classification",
        "",
        f"`{result['classification']}`",
        "",
        f"Exactly one next experiment: {result['next_experiment']}",
        "",
        "## Baseline Reproduction",
        "",
        "| seed | reference SHA | instrumented SHA | parity |",
        "| --- | --- | --- | --- |",
    ]
    for seed, row in result["baseline_reproduction"].items():
        lines.append(
            f"| {seed} | `{row['reference_sha256']}` | `{row['instrumented_sha256']}` | {row['reproduced']} |"
        )
    lines += [
        "",
        "## Exact Loss",
        "",
        "`L_total = L_policy + 0.3 * L_value`; pairwise and behavior-anchor weights were asserted zero on every live minibatch. The machine-readable artifact records pre/post-common-clip component norms, additive identity errors, residual-block cosines, anchor/cluster first-order effects, actual Adam-update alignment, and chronological validation history.",
        "",
        "## Shared-Trunk Attribution",
        "",
        "| seed | policy harmful anchor | value harmful anchor | policy helpful | value helpful |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for seed, row in result["contributions"].items():
        lines.append(
            f"| {seed} | {row['policy_harmful']:.6g} | {row['value_harmful']:.6g} | {row['policy_helpful']:.6g} | {row['value_helpful']:.6g} |"
        )
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
        raise RuntimeError("frozen cluster identity/injection guard failed")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "artifacts": artifacts,
        "g0_sha256": G0_SHA,
        "guardrails": {
            "self_play": False,
            "replay_mutation": False,
            "training_intervention": False,
            "promotion": False,
            "exact_labels_as_training_targets": False,
        },
    }
    if not args.execute:
        result |= {"classification": "planned", "next_experiment": "not run"}
        write_json(args.out_result, result)
        args.out_report.write_text(
            "# R61 Policy vs Value Shared-Trunk Gradient Audit\n\nPlanned diagnostic only.\n"
        )
        return 0
    lanes = {}
    baseline = {}
    for seed in TRAINING_SEEDS:
        reference, live = (
            run_lane(paths, manifest, seed, args.workdir, False),
            run_lane(paths, manifest, seed, args.workdir, True),
        )
        parity = (
            reference["checkpoint_sha256"]
            == live["checkpoint_sha256"]
            == EXPECTED_SHA[seed]
            and reference["permutations"] == live["permutations"]
            and [row["anchor_margin_after"] for row in reference["traces"]]
            == [row["anchor_margin_after"] for row in live["traces"]]
            and reference["epoch_history"] == live["epoch_history"]
        )
        baseline[seed] = {
            "expected_sha256": EXPECTED_SHA[seed],
            "reference_sha256": reference["checkpoint_sha256"],
            "instrumented_sha256": live["checkpoint_sha256"],
            "permutations": live["permutations"],
            "selected_best_epoch": min(
                live["epoch_history"],
                key=lambda row: float(row["validation_total_loss"]),
            )["epoch"],
            "reproduced": parity,
        }
        lanes[seed] = live
    if not all(row["reproduced"] for row in baseline.values()):
        result |= {
            "baseline_reproduction": baseline,
            "classification": "loss_gradient_attribution_baseline_not_reproduced",
            "next_experiment": "none",
        }
        write_json(args.out_result, result)
        raise RuntimeError("loss_gradient_attribution_baseline_not_reproduced")
    contributions = {
        seed: contribution(lane["traces"], "shared_trunk", "anchor_effect")
        for seed, lane in lanes.items()
    }
    t61_steps = lanes["T61"]["traces"]
    largest_actual = sorted(
        range(len(t61_steps)),
        key=lambda index: (
            t61_steps[index]["anchor_margin_after"]
            - t61_steps[index]["anchor_margin_before"]
        ),
    )[:10]
    largest_predicted = sorted(
        range(len(t61_steps)),
        key=lambda index: min(
            t61_steps[index]["groups"]["shared_trunk"]["policy_anchor_effect"],
            t61_steps[index]["groups"]["shared_trunk"]["value_anchor_effect"],
        ),
    )[:10]
    selected = sorted(set(largest_actual) | set(largest_predicted))
    counterfactuals = {
        seed: counterfactual_summary(
            lanes[seed],
            lanes[seed]["compact_x"],
            lanes[seed]["compact_p"],
            lanes[seed]["compact_v"],
            manifest,
            selected,
        )
        for seed in TRAINING_SEEDS
    }
    cluster_contributions = {
        seed: contribution(lane["traces"], "shared_trunk", "cluster_effect")
        for seed, lane in lanes.items()
    }
    classification, next_experiment = classify(
        contributions["T61"],
        contributions["T63"],
        cluster_contributions["T61"],
        cluster_contributions["T63"],
        counterfactuals["T61"],
    )
    result |= {
        "baseline_reproduction": baseline,
        "parameter_groups": lanes["T61"]["groups"],
        "loss_decomposition": {
            "policy": 1.0,
            "value": VALUE_WEIGHT,
            "pairwise": 0.0,
            "behavior_anchor": 0.0,
        },
        "trajectories": {
            seed: {
                "step_trace": lane["traces"],
                "epoch_history": lane["epoch_history"],
                "aggregate": {
                    group: aggregate(lane["traces"], group)
                    for group in (*FINE_GROUPS, "shared_trunk")
                },
            }
            for seed, lane in lanes.items()
        },
        "contributions": contributions,
        "cluster_contributions": cluster_contributions,
        "selected_counterfactual_step_indexes": selected,
        "adam_counterfactuals": counterfactuals,
        "classification": classification,
        "next_experiment": next_experiment,
    }
    write_json(args.out_result, result)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
