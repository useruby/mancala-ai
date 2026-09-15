#!/usr/bin/env python3
"""Diagnostic-only activation localization for the frozen R61 T61/T63 split.

The live observer evaluates frozen states after optimizer steps under ``eval`` /
``no_grad`` only. It never changes replay, losses, model parameters, or search.
Activation transplants are offline continuations through recipient layers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from ml.alphazero_lite.run_g1_replay_optimizer_crossover import (
    ANCHOR_ID,
    FROZEN_SET_SHA,
    G0_SHA,
    artifact_paths,
    sha256_file,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import MANIFEST_SCHEMA
from ml.alphazero_lite.run_r61_parameter_subspace_drift_audit import (
    make_hybrid,
    parameter_groups,
)
from ml.alphazero_lite.run_r61_lr_stability_ablation import (
    REPLAY,
    REPLAY_WEIGHTS,
    verify_r61_artifacts,
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

SCHEMA = "azlite_r61_activation_representation_audit_v1"
TRAINING_SEEDS = {"T61": 61, "T63": 63}
EXPECTED_SHA = {
    "T61": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "T63": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
}
STAGES = ("A0", "A1", "A2", "A3", "A4", "A5")
PATCH_STAGES = STAGES[:-1]
EPS = 1e-12
TOLERANCE = 2e-6
EPOCHS = 4


def snapshot(model: PolicyValueNet) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }


def restore(model: PolicyValueNet, state: dict[str, torch.Tensor]) -> PolicyValueNet:
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(state[name])
    return model


def residual_v3_activations(
    model: PolicyValueNet, x: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Expose actual residual_v3 graph nodes, not an architectural approximation."""
    if model.model_type != "residual_v3":
        raise ValueError("activation audit requires residual_v3")
    assert model.input_layer is not None and model.policy_hidden_layer is not None
    h = torch.relu(model.input_layer(x))
    output = {"A0": h}
    for index, (first, second) in enumerate(model.residual_layers):
        residual = h
        h = torch.relu(first(h))
        h = torch.relu(second(h) + residual)
        output[f"A{index + 1}"] = h
    output["A4"] = torch.relu(model.policy_hidden_layer(h))
    output["A5"] = model.policy_head(output["A4"])
    if tuple(output) != STAGES:
        raise RuntimeError("unexpected residual_v3 activation stages")
    return output


def continue_policy_from_stage(
    model: PolicyValueNet, stage: str, activation: torch.Tensor
) -> torch.Tensor:
    """Continue through recipient policy layers after an exposed graph node."""
    if stage not in PATCH_STAGES:
        raise ValueError(f"stage is not patchable: {stage}")
    assert model.policy_hidden_layer is not None
    h = activation
    if stage == "A0":
        start = 0
    elif stage in {"A1", "A2", "A3"}:
        start = int(stage[1:])
    else:  # A4 is already the policy hidden ReLU output.
        return model.policy_head(h)
    for first, second in model.residual_layers[start:]:
        residual = h
        h = torch.relu(first(h))
        h = torch.relu(second(h) + residual)
    h = torch.relu(model.policy_hidden_layer(h))
    return model.policy_head(h)


def block_output(
    model: PolicyValueNet, block: int, incoming: torch.Tensor
) -> torch.Tensor:
    first, second = model.residual_layers[block]
    return torch.relu(second(torch.relu(first(incoming))) + incoming)


def vector_metrics(value: torch.Tensor, g0: torch.Tensor) -> dict[str, float]:
    value, g0 = (
        value.detach().float().reshape(-1).cpu(),
        g0.detach().float().reshape(-1).cpu(),
    )
    norm = float(torch.linalg.vector_norm(value))
    base_norm = float(torch.linalg.vector_norm(g0))
    distance = float(torch.linalg.vector_norm(value - g0))
    cosine = float(torch.dot(value, g0) / max(norm * base_norm, EPS))
    return {
        "norm": norm,
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "inactive_fraction": float((value <= 1e-8).float().mean()),
        "cosine_to_g0": cosine,
        "normalized_l2_to_g0": distance / max(base_norm, EPS),
    }


def pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left, right = (
        left.detach().float().reshape(-1).cpu(),
        right.detach().float().reshape(-1).cpu(),
    )
    ln, rn = (
        float(torch.linalg.vector_norm(left)),
        float(torch.linalg.vector_norm(right)),
    )
    distance = float(torch.linalg.vector_norm(left - right))
    return {
        "cosine": float(torch.dot(left, right) / max(ln * rn, EPS)),
        "normalized_l2": distance / (0.5 * (ln + rn) + EPS),
        "distance": distance,
        "support_disagreement": float(((left > 1e-8) != (right > 1e-8)).float().mean()),
    }


def legal_policy_metrics(logits: torch.Tensor, entry: dict[str, Any]) -> dict[str, Any]:
    legal = entry["legal_actions"]
    values = logits.detach().cpu()[0]
    policy = torch.softmax(values[legal], dim=0)
    by_action = {action: float(policy[index]) for index, action in enumerate(legal)}
    top = legal[int(torch.argmax(policy))]
    optimal = entry["exact_outcome_optimal_actions"]
    other = max(float(values[a]) for a in legal if a not in optimal)
    best_optimal = max(float(values[a]) for a in optimal)
    return {
        "top_action": top,
        "top_is_outcome_optimal": top in optimal,
        "optimal_mass": sum(by_action[a] for a in optimal),
        "margin": best_optimal - other,
        "policy": by_action,
    }


def patch_metrics(
    recipient: PolicyValueNet,
    donor: PolicyValueNet,
    stage: str,
    x: torch.Tensor,
    entry: dict[str, Any],
) -> dict[str, Any]:
    with torch.no_grad():
        activation = residual_v3_activations(donor, x)[stage]
        return legal_policy_metrics(
            continue_policy_from_stage(recipient, stage, activation), entry
        )


def rescue(native_t61: dict[str, Any], patched: dict[str, Any]) -> bool:
    return (
        not native_t61["top_is_outcome_optimal"] and patched["top_is_outcome_optimal"]
    )


def reverse_break(native_t63: dict[str, Any], patched: dict[str, Any]) -> bool:
    return (
        native_t63["top_is_outcome_optimal"] and not patched["top_is_outcome_optimal"]
    )


def select_earliest_causal_stage(rows: dict[str, dict[str, Any]]) -> str | None:
    for stage in PATCH_STAGES:
        value = rows[stage]
        if (
            value["anchor_rescue"]
            and value["anchor_reverse_harm"]
            and value["rescue_rate"] >= 0.60
            and value["mean_transfer_fraction"] >= 0.60
            and value["control_degradation"] < 0.20
        ):
            return stage
    return None


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None

    # Stable average-rank implementation without a scipy dependency.
    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        rank = np.empty(len(values), dtype=float)
        rank[order] = np.arange(len(values), dtype=float)
        for value in set(values):
            indexes = np.where(np.asarray(values) == value)[0]
            rank[indexes] = rank[indexes].mean()
        return rank

    a, b = ranks(left), ranks(right)
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None


def state_inputs(manifest: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        row["id"]: torch.tensor(
            [encode_state(row["state"], input_encoding="kalah_v3")], dtype=torch.float32
        )
        for row in manifest["entries"]
    }


def evaluate_activations(
    model: PolicyValueNet, inputs: dict[str, torch.Tensor], g0: PolicyValueNet
) -> dict[str, Any]:
    old = model.training
    model.eval()
    with torch.no_grad():
        output = {
            ident: {
                stage: vector_metrics(
                    value, residual_v3_activations(g0, inputs[ident])[stage]
                )
                for stage, value in residual_v3_activations(
                    model, inputs[ident]
                ).items()
            }
            for ident in inputs
        }
    model.train(old)
    return output


def checkpoint(model: PolicyValueNet, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **checkpoint_from_model(model))
    return sha256_file(path)


def run_lane(
    paths: dict[str, Any],
    manifest: dict[str, Any],
    seed: str,
    workdir: Path,
    instrumented: bool,
) -> dict[str, Any]:
    set_seed(TRAINING_SEEDS[seed])
    x, p, v, indexes = load_jsonl_replay(
        [paths["replays"][REPLAY], *paths["fixed"]],
        list(REPLAY_WEIGHTS),
        policy_target_mode="sharpened",
        value_target_mode="sharpened",
    )
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, paths["parent"])
    g0 = copy.deepcopy(model).eval()
    inputs, telemetry, states, permutations, history = (
        state_inputs(manifest),
        [],
        {0: snapshot(model)},
        {},
        [],
    )
    # Retain post-step parameter snapshots temporarily so percentage landmarks
    # are selected from the exact, rather than estimated, historical cadence.
    steps = 0

    def permutation(epoch: int | None, values: list[int]) -> None:
        permutations[str(epoch)] = hashlib.sha256(
            json.dumps(values).encode()
        ).hexdigest()

    def callback(phase: str, context: dict[str, Any]) -> None:
        nonlocal steps
        if phase != "after":
            return
        steps += 1
        step = steps
        if instrumented:
            telemetry.append(
                {
                    "optimizer_step": step,
                    "epoch": context["epoch"],
                    "states": evaluate_activations(model, inputs, g0),
                }
            )
        states[step] = snapshot(model)

    def epoch_callback(
        _epoch: int, _optimizer: torch.optim.Optimizer, current: torch.nn.Module
    ) -> None:
        states[steps] = snapshot(current)

    train(
        model,
        x,
        p,
        v,
        indexes,
        epochs=EPOCHS,
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
        step_callback=callback,
        epoch_callback=epoch_callback,
        permutation_callback=permutation,
        epoch_history=history,
    )
    total_steps = steps
    states[total_steps] = snapshot(model)
    selected = {0, 1, 2, 3, 4, total_steps}
    selected |= {round(total_steps * fraction) for fraction in (0.25, 0.5, 0.75)}
    selected |= {round(total_steps * epoch / EPOCHS) for epoch in range(1, EPOCHS + 1)}
    states = {step: value for step, value in states.items() if step in selected}
    path = workdir / ("instrumented" if instrumented else "reference") / f"{seed}.npz"
    return {
        "model": model,
        "g0": g0,
        "telemetry": telemetry,
        "states": states,
        "permutations": permutations,
        "history": history,
        "sha": checkpoint(model, path),
        "path": path,
        "total_steps": total_steps,
    }


def patch_matrix(
    t61: PolicyValueNet,
    t63: PolicyValueNet,
    manifest: dict[str, Any],
    inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    entries = manifest["entries"]
    native = {
        name: {
            row["id"]: legal_policy_metrics(
                residual_v3_activations(model, inputs[row["id"]])["A5"], row
            )
            for row in entries
        }
        for name, model in (("T61", t61), ("T63", t63))
    }
    output: dict[str, Any] = {"native": native, "stages": {}}
    for stage in PATCH_STAGES:
        rows, disagreement, controls = [], [], []
        for row in entries:
            ident = row["id"]
            forward, reverse = (
                patch_metrics(t61, t63, stage, inputs[ident], row),
                patch_metrics(t63, t61, stage, inputs[ident], row),
            )
            item = {
                "id": ident,
                "forward": forward,
                "reverse": reverse,
                "rescue": rescue(native["T61"][ident], forward),
                "reverse_break": reverse_break(native["T63"][ident], reverse),
            }
            rows.append(item)
            if row["membership"] == "matched_control":
                controls.append(item)
            elif (
                not native["T61"][ident]["top_is_outcome_optimal"]
                and native["T63"][ident]["top_is_outcome_optimal"]
            ):
                disagreement.append(item)
        anchor = next(item for item in rows if item["id"] == ANCHOR_ID)
        fractions = [
            (item["forward"]["margin"] - native["T61"][item["id"]]["margin"])
            / (
                native["T63"][item["id"]]["margin"]
                - native["T61"][item["id"]]["margin"]
            )
            for item in disagreement
            if native["T63"][item["id"]]["margin"]
            != native["T61"][item["id"]]["margin"]
        ]
        output["stages"][stage] = {
            "rows": rows,
            "anchor_rescue": anchor["rescue"],
            "anchor_reverse_harm": anchor["reverse_break"]
            or anchor["reverse"]["margin"] < native["T63"][ANCHOR_ID]["margin"],
            "rescue_rate": float(np.mean([item["rescue"] for item in disagreement]))
            if disagreement
            else 0.0,
            "mean_transfer_fraction": float(np.mean(fractions)) if fractions else 0.0,
            "control_degradation": float(
                np.mean([item["reverse_break"] for item in controls])
            )
            if controls
            else 0.0,
        }
    output["earliest_causal_stage"] = select_earliest_causal_stage(output["stages"])
    return output


def validate_activation_helpers(model: PolicyValueNet, x: torch.Tensor) -> None:
    with torch.no_grad():
        activations = residual_v3_activations(model, x)
        logits, _ = model(x)
        if not torch.allclose(
            activations["A3"], model.trunk_features(x), atol=TOLERANCE, rtol=0
        ):
            raise RuntimeError("activation_a3_parity_failed")
        if not torch.allclose(activations["A5"], logits, atol=TOLERANCE, rtol=0):
            raise RuntimeError("activation_logit_parity_failed")
        for stage in PATCH_STAGES:
            if not torch.allclose(
                continue_policy_from_stage(model, stage, activations[stage]),
                logits,
                atol=TOLERANCE,
                rtol=0,
            ):
                raise RuntimeError("activation_self_patch_failed")


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
        raise RuntimeError("frozen_manifest_guard_failed")
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
        args.out_result.parent.mkdir(parents=True, exist_ok=True)
        args.out_result.write_text(json.dumps(result, indent=2) + "\n")
        return 0
    lanes, baseline = {}, {}
    for seed in TRAINING_SEEDS:
        reference, live = (
            run_lane(paths, manifest, seed, args.workdir, False),
            run_lane(paths, manifest, seed, args.workdir, True),
        )
        parity = (
            reference["sha"] == live["sha"] == EXPECTED_SHA[seed]
            and reference["permutations"] == live["permutations"]
            and reference["history"] == live["history"]
        )
        baseline[seed] = {
            "expected_sha256": EXPECTED_SHA[seed],
            "reference_sha256": reference["sha"],
            "instrumented_sha256": live["sha"],
            "reproduced": parity,
        }
        lanes[seed] = live
    if not all(row["reproduced"] for row in baseline.values()):
        result |= {
            "baseline_reproduction": baseline,
            "classification": "activation_audit_baseline_not_reproduced",
            "next_experiment": "none",
        }
        args.out_result.write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError("activation_audit_baseline_not_reproduced")
    inputs = state_inputs(manifest)
    for model in (lanes["T61"]["model"], lanes["T63"]["model"]):
        for value in inputs.values():
            validate_activation_helpers(model, value)
    patches = patch_matrix(
        lanes["T61"]["model"], lanes["T63"]["model"], manifest, inputs
    )
    # A3 is exactly the endpoint shared-trunk transplant from #315.
    groups = parameter_groups(lanes["T61"]["model"])
    for recipient, donor, direction in (
        ("T61", "T63", "forward"),
        ("T63", "T61", "reverse"),
    ):
        hybrid = make_hybrid(
            lanes[recipient]["model"], lanes[donor]["model"], groups, ("shared_trunk",)
        )
        for row in manifest["entries"]:
            with torch.no_grad():
                patched_logits = continue_policy_from_stage(
                    lanes[recipient]["model"],
                    "A3",
                    residual_v3_activations(lanes[donor]["model"], inputs[row["id"]])[
                        "A3"
                    ],
                )
                expected_logits = hybrid(inputs[row["id"]])[0]
            if not torch.allclose(
                patched_logits, expected_logits, atol=TOLERANCE, rtol=0
            ):
                raise RuntimeError("activation_patching_invalid")
    stage = patches["earliest_causal_stage"]
    classification = {
        "A0": "cluster_representation_drift_early_trunk_primary",
        "A1": "cluster_representation_drift_early_trunk_primary",
        "A2": "cluster_representation_drift_mid_trunk_primary",
        "A3": "cluster_representation_drift_late_trunk_primary",
        "A4": "cluster_representation_drift_policy_amplified",
    }.get(stage, "representation_drift_distributed")
    next_experiment = "perform a replay-minibatch provenance audit using final-trunk cluster activation drift as the response, rather than another parameter/scope intervention."
    result |= {
        "baseline_reproduction": baseline,
        "activation_telemetry": {
            seed: lane["telemetry"] for seed, lane in lanes.items()
        },
        "final_patching": patches,
        "classification": classification,
        "next_experiment": next_experiment,
    }
    args.out_result.parent.mkdir(parents=True, exist_ok=True)
    args.out_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        "# R61 Activation Representation Audit\n\nInherited #315: `anchor_drift_shared_trunk_primary`; #316: `heads_only_no_anchor_rescue`; #317: `last_block_policy_no_anchor_rescue`; #318: `shared_trunk_loss_gradient_not_explanatory`.\n\n## Classification\n\n`"
        + classification
        + "`\n\nExactly one next experiment: "
        + next_experiment
        + "\n\n## Baseline\n\n"
        + json.dumps(baseline, indent=2)
        + "\n\nCompact per-step activation telemetry and complete offline patch matrix are in the machine-readable JSON artifact.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
