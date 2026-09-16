#!/usr/bin/env python3
"""Five-step R61 parameter-versus-Adam recipient-state decomposition."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.run_g1_replay_optimizer_crossover import (
    ANCHOR_ID,
    FROZEN_SET_SHA,
    artifact_paths,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import MANIFEST_SCHEMA
from ml.alphazero_lite.run_r61_activation_representation_audit import (
    continue_policy_from_stage,
    legal_policy_metrics,
    residual_v3_activations,
)
from ml.alphazero_lite.run_r61_earliest_a0_divergence_audit import (
    CLONE_TOLERANCE,
    adam_update,
    anchor_effect,
    max_tensor_error,
    reconstruct_lane,
)
from ml.alphazero_lite.run_r61_early_trunk_replay_provenance_audit import (
    cohort,
    cosine,
    evaluate_step,
    input_vector,
    model_from_snapshot,
    tensor_snapshot,
    verify_r61_artifacts,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import PolicyValueNet, load_checkpoint_into_model

SCHEMA = "azlite_r61_step82_parameter_adam_decomposition_v1"
STEPS = (80, 81, 82, 83, 84)
INPUT_NAMES = ("input_layer.weight", "input_layer.bias")
CELL_SPECS = {
    "PP61_A61": ("T61", "T61"),
    "P61_A63": ("T61", "T63"),
    "P63_A61": ("T63", "T61"),
    "P63_A63": ("T63", "T63"),
}


def write_json(path: Path, value: Any) -> None:
    """Write a deterministic machine-readable diagnostic artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_optimizer_compatibility(
    parameters: dict[str, torch.Tensor], optimizer: dict[str, Any]
) -> None:
    """Reject a moment transplant unless all historical parameter slots align."""
    names = tuple(parameters)
    slots = [slot for group in optimizer["param_groups"] for slot in group["params"]]
    if len(slots) != len(names) or len(set(slots)) != len(slots):
        raise RuntimeError("optimizer_parameter_mapping_invalid")
    for name, slot in zip(names, slots, strict=True):
        state = optimizer["state"].get(slot, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and tuple(state[key].shape) != tuple(
                parameters[name].shape
            ):
                raise RuntimeError("optimizer_parameter_mapping_invalid")


def transplant_parameters(
    recipient: dict[str, torch.Tensor], donor: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Deep-copy a full model parameter state from the donor."""
    if recipient.keys() != donor.keys() or any(
        recipient[name].shape != donor[name].shape for name in recipient
    ):
        raise RuntimeError("parameter_transplant_incompatible")
    return {name: value.detach().clone() for name, value in donor.items()}


def transplant_optimizer_state(
    recipient: dict[str, Any],
    donor: dict[str, Any],
    parameters: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Deep-copy an Adam state after validating it against recipient parameter order."""
    result = copy.deepcopy(donor)
    validate_optimizer_compatibility(parameters, result)
    return result


def transplant_input_parameters(
    recipient: dict[str, torch.Tensor], donor: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Deep-copy recipient parameters with only the complete input layer replaced."""
    result = {name: value.detach().clone() for name, value in recipient.items()}
    for name in INPUT_NAMES:
        if result[name].shape != donor[name].shape:
            raise RuntimeError("input_parameter_transplant_incompatible")
        result[name] = donor[name].detach().clone()
    return result


def transplant_input_adam_state(
    recipient: dict[str, Any],
    donor: dict[str, Any],
    parameters: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Deep-copy Adam state with only input-layer moments and step replaced."""
    result = copy.deepcopy(recipient)
    validate_optimizer_compatibility(parameters, result)
    slots = [slot for group in result["param_groups"] for slot in group["params"]]
    donor_slots = [slot for group in donor["param_groups"] for slot in group["params"]]
    for name, slot, donor_slot in zip(parameters, slots, donor_slots, strict=True):
        if name in INPUT_NAMES:
            result["state"][slot] = copy.deepcopy(donor["state"][donor_slot])
    validate_optimizer_compatibility(parameters, result)
    return result


def factorial(cells: dict[str, float]) -> dict[str, float]:
    """Calculate the registered 2x2 parameter, Adam, and interaction effects."""
    a, b, c, d = (cells[name] for name in ("PP61_A61", "P61_A63", "P63_A61", "P63_A63"))
    values = {
        "parameter_main_effect": 0.5 * ((c - a) + (d - b)),
        "adam_main_effect": 0.5 * ((b - a) + (d - c)),
        "interaction": (d - c) - (b - a),
    }
    return values | {f"abs_{name}": abs(value) for name, value in values.items()}


def response_metrics(
    before: dict[str, torch.Tensor],
    after: PolicyValueNet,
    entries: list[dict[str, Any]],
    input_size: int,
) -> dict[str, Any]:
    """Evaluate one local update using the inherited cluster/control/anchor semantics."""
    before_model = model_from_snapshot(before, input_size)
    rows, effects = evaluate_step(before_model, after, entries)
    cluster_margins, anchor_margin = [], None
    for entry in entries:
        x = torch.tensor(
            [encode_state(entry["state"], input_encoding="kalah_v3")],
            dtype=torch.float32,
        )
        margin = legal_policy_metrics(residual_v3_activations(after, x)["A5"], entry)[
            "margin"
        ]
        if cohort(entry) == "cluster":
            cluster_margins.append(float(margin))
        if entry["id"] == ANCHOR_ID:
            anchor_margin = float(margin)
    movements = [
        float(row["movement"]["l2"]) for row in rows if row["cohort"] == "cluster"
    ]
    return effects | {
        "anchor_a0_only_margin_effect": anchor_effect(rows),
        "cluster_mean_margin": statistics.fmean(cluster_margins),
        "anchor_margin": anchor_margin,
        "input_layer_update_norm": float(
            torch.linalg.vector_norm(
                input_vector(tensor_snapshot(after)) - input_vector(before)
            )
        ),
        "a0_representation_movement": statistics.fmean(movements),
        "support_flips": sum(
            int(row["movement"]["support_flips"])
            for row in rows
            if row["cohort"] == "cluster"
        ),
        "state_rows": rows,
    }


def run_sequence(
    initial: dict[str, torch.Tensor],
    optimizer: dict[str, Any],
    lane: dict[str, Any],
    batches: list[list[int]],
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], PolicyValueNet]:
    """Apply exactly five immutable historical batches to an independently cloned state."""
    model, state, rows = (
        model_from_snapshot(initial, lane["x"].shape[1]),
        copy.deepcopy(optimizer),
        [],
    )
    for step, indexes in zip(STEPS, batches, strict=True):
        before = tensor_snapshot(model)
        model, updated = adam_update(
            before, state, lane["x"][indexes], lane["p"][indexes], lane["v"][indexes]
        )
        state = updated.state_dict()
        rows.append(
            {
                "step": step,
                "batch_indexes": list(indexes),
                **response_metrics(before, model, entries, lane["x"].shape[1]),
            }
        )
    return rows, model


def final_response(
    initial: dict[str, torch.Tensor],
    final: PolicyValueNet,
    entries: list[dict[str, Any]],
    input_size: int,
) -> dict[str, float]:
    """Summarize the registered five-step endpoint without continuing training."""
    initial_model = model_from_snapshot(initial, input_size)
    rows, _ = evaluate_step(initial_model, final, entries)
    final_metrics = response_metrics(initial, final, entries, input_size)
    return {
        "cumulative_cluster_specific_a0_movement": float(
            sum(
                row["a0_only_margin_delta"]
                for row in rows
                if row["cohort"] == "cluster"
            )
            - sum(
                row["a0_only_margin_delta"]
                for row in rows
                if row["cohort"] == "controls"
            )
        ),
        "final_cluster_mean_margin": final_metrics["cluster_mean_margin"],
        "final_anchor_margin": final_metrics["anchor_margin"],
        "final_a0_distance_from_own_initial_state": final_metrics[
            "a0_representation_movement"
        ],
    }


def downstream_fixed_context(
    recipient: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor],
    entries: list[dict[str, Any]],
    input_size: int,
) -> dict[str, float]:
    """Evaluate a donor input layer through unchanged recipient downstream layers."""
    native = model_from_snapshot(recipient, input_size)
    hybrid = model_from_snapshot(
        transplant_input_parameters(recipient, donor), input_size
    )
    cluster, anchor = [], None
    for entry in entries:
        x = torch.tensor(
            [encode_state(entry["state"], input_encoding="kalah_v3")],
            dtype=torch.float32,
        )
        delta = float(
            legal_policy_metrics(residual_v3_activations(hybrid, x)["A5"], entry)[
                "margin"
            ]
            - legal_policy_metrics(residual_v3_activations(native, x)["A5"], entry)[
                "margin"
            ]
        )
        if cohort(entry) == "cluster":
            cluster.append(delta)
        if entry["id"] == ANCHOR_ID:
            anchor = delta
    return {
        "cluster_mean_margin_delta": statistics.fmean(cluster),
        "anchor_margin_delta": anchor,
    }


def input_geometry(
    t61: dict[str, torch.Tensor],
    t63: dict[str, torch.Tensor],
    g0: dict[str, torch.Tensor],
    a61: dict[str, Any],
    a63: dict[str, Any],
) -> dict[str, Any]:
    """Report pre-step-80 input parameter and Adam geometry."""

    def vector(values: dict[str, torch.Tensor]) -> torch.Tensor:
        return input_vector(values)

    def moments(state: dict[str, Any], key: str) -> torch.Tensor:
        slots = [slot for group in state["param_groups"] for slot in group["params"]]
        return torch.cat(
            [state["state"][slot][key].reshape(-1).float() for slot in slots[:2]]
        )

    p61, p63, pg0 = vector(t61), vector(t63), vector(g0)
    m61, m63, v61, v63 = (
        moments(a61, "exp_avg"),
        moments(a63, "exp_avg"),
        moments(a61, "exp_avg_sq"),
        moments(a63, "exp_avg_sq"),
    )
    return {
        "parameters": {
            "l2_distance": float(torch.linalg.vector_norm(p61 - p63)),
            "drift_from_g0_cosine": cosine(p61 - pg0, p63 - pg0),
        },
        "adam_first_moments": {
            "l2_distance": float(torch.linalg.vector_norm(m61 - m63)),
            "cosine": cosine(m61, m63),
        },
        "adam_second_moments": {
            "l2_distance": float(torch.linalg.vector_norm(v61 - v63)),
            "relative_scale_ratio": float(
                torch.linalg.vector_norm(v63)
                / max(torch.linalg.vector_norm(v61), torch.tensor(1e-12))
            ),
        },
        "a0_units": [
            {
                "unit": unit,
                "row_weight_distance": float(
                    torch.linalg.vector_norm(
                        t61["input_layer.weight"][unit]
                        - t63["input_layer.weight"][unit]
                    )
                ),
                "bias_difference": float(
                    t63["input_layer.bias"][unit] - t61["input_layer.bias"][unit]
                ),
            }
            for unit in range(t61["input_layer.weight"].shape[0])
        ],
    }


def unit_localization(
    t61: dict[str, torch.Tensor],
    t63: dict[str, torch.Tensor],
    entries: list[dict[str, Any]],
    input_size: int,
) -> list[dict[str, Any]]:
    """Rank A0 units observationally; the downstream gradient is a first-order probe."""
    m61, m63 = (
        model_from_snapshot(t61, input_size),
        model_from_snapshot(t63, input_size),
    )
    result = []
    for unit in range(t61["input_layer.weight"].shape[0]):
        cluster, controls, flips, anchor = [], [], 0, []
        for entry in entries:
            x = torch.tensor(
                [encode_state(entry["state"], input_encoding="kalah_v3")],
                dtype=torch.float32,
            )
            a61, a63 = (
                residual_v3_activations(m61, x)["A0"],
                residual_v3_activations(m63, x)["A0"],
            )
            delta = float((a63[0, unit] - a61[0, unit]).detach())
            if cohort(entry) == "cluster":
                cluster.append(abs(delta))
                flips += int((a61[0, unit] > 0) != (a63[0, unit] > 0))
            if cohort(entry) == "controls":
                controls.append(delta)
            if entry["id"] == ANCHOR_ID:
                probe = a61.detach().clone().requires_grad_(True)
                logits = continue_policy_from_stage(m61, "A0", probe)
                legal = entry["legal_actions"]
                target = entry["exact_outcome_optimal_actions"][0]
                alternatives = [action for action in legal if action != target]
                (logits[0, target] - torch.max(logits[0, alternatives])).backward()
                anchor.append(
                    float(probe.grad[0, unit] * (a63[0, unit] - a61[0, unit]).detach())
                )
        result.append(
            {
                "unit": unit,
                "cluster_mean_absolute_delta": statistics.fmean(cluster),
                "matched_control_delta": statistics.fmean(controls)
                if controls
                else 0.0,
                "relu_support_flip_frequency": flips / max(len(cluster), 1),
                "anchor_margin_first_order_contribution": statistics.fmean(anchor)
                if anchor
                else 0.0,
            }
        )
    return sorted(
        result, key=lambda row: (-row["cluster_mean_absolute_delta"], row["unit"])
    )[:10]


def classify(per_step: list[dict[str, Any]], finals: dict[str, Any]) -> tuple[str, str]:
    """Apply the hard dominance decision tree to A0 and cluster-margin responses."""
    step82 = next(row for row in per_step if row["step"] == 82)
    parameter_wins = sum(
        row["factorial"]["abs_parameter_main_effect"]
        > row["factorial"]["abs_adam_main_effect"]
        for row in per_step
    )
    final_a0, final_margin = finals["a0_factorial"], finals["margin_factorial"]
    parameter = (
        step82["factorial"]["abs_parameter_main_effect"]
        > step82["factorial"]["abs_adam_main_effect"]
        and parameter_wins >= 3
        and final_a0["abs_parameter_main_effect"]
        >= 1.5 * final_a0["abs_adam_main_effect"]
        and final_margin["abs_parameter_main_effect"]
        > final_margin["abs_adam_main_effect"]
    )
    if parameter:
        return (
            "early_divergence_parameter_representation_primary",
            "trace which INPUT-LAYER parameter directions / A0 units create the cluster-specific divergence immediately before step 80–82, using the unit-level localization from this audit.",
        )
    return (
        "early_divergence_local_decomposition_mixed",
        "move to the input-layer/A0 unit-level representation geometry at steps 80–82 rather than doing another optimizer-state sweep.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-result", type=Path, required=True)
    parser.add_argument("--out-steps", type=Path, required=True)
    parser.add_argument("--out-input-localization", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "inherited_classification": "earliest_divergence_mixed",
        "steps": list(STEPS),
        "guardrails": {
            "replay_mutation": False,
            "self_play": False,
            "full_training": False,
            "promotion": False,
        },
    }
    if not args.execute:
        write_json(args.out_result, result | {"classification": "planned"})
        return 0
    paths = artifact_paths()
    verify_r61_artifacts(paths)
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
    lanes = {
        seed: reconstruct_lane(paths, manifest, seed, set(STEPS), args.workdir)
        for seed in ("T61", "T63")
    }
    native: dict[str, bool] = {}
    # Clone replay validates every native transition and the step-84 endpoint.
    for seed in lanes:
        before = lanes[seed]["snapshots"][80]["before"]
        optimizer = lanes[seed]["snapshots"][80]["optimizer"]
        valid = True
        for step in STEPS:
            indexes = lanes[seed]["trace"][step - 1]["batch_indexes"]
            model, updated = adam_update(
                before,
                optimizer,
                lanes[seed]["x"][indexes],
                lanes[seed]["p"][indexes],
                lanes[seed]["v"][indexes],
            )
            valid &= (
                max_tensor_error(model, lanes[seed]["snapshots"][step]["after"])
                <= CLONE_TOLERANCE
            )
            before, optimizer = tensor_snapshot(model), updated.state_dict()
        native[seed] = valid
    if not all(native.values()):
        raise RuntimeError("local_state_decomposition_invalid")
    snapshots = {seed: lanes[seed]["snapshots"][80] for seed in lanes}
    batches = {
        seed: [lanes[seed]["trace"][step - 1]["batch_indexes"] for step in STEPS]
        for seed in lanes
    }
    all_runs: dict[str, dict[str, Any]] = {}
    for batch_name, batch_sequence in batches.items():
        runs = {}
        for name, (p_seed, a_seed) in CELL_SPECS.items():
            params = transplant_parameters(
                snapshots[p_seed]["before"], snapshots[p_seed]["before"]
            )
            optimizer = transplant_optimizer_state(
                snapshots[a_seed]["optimizer"], snapshots[a_seed]["optimizer"], params
            )
            rows, model = run_sequence(
                params,
                optimizer,
                lanes[batch_name],
                batch_sequence,
                manifest["entries"],
            )
            final = final_response(
                params, model, manifest["entries"], lanes[batch_name]["x"].shape[1]
            )
            final["cumulative_cluster_specific_a0_movement"] = sum(
                float(row["cluster_specific_a0_effect"]) for row in rows
            )
            runs[name] = {"per_step": rows, "final": final}
        all_runs[batch_name] = runs
    primary = all_runs["T61"]
    per_step = [
        {
            "step": step,
            "cells": {
                name: next(
                    row for row in primary[name]["per_step"] if row["step"] == step
                )
                for name in CELL_SPECS
            },
        }
        for step in STEPS
    ]
    for row in per_step:
        row["factorial"] = factorial(
            {
                name: cell["cluster_specific_a0_effect"]
                for name, cell in row["cells"].items()
            }
        )
    compact_per_step = [
        {
            "step": row["step"],
            "batch_indexes_sha256": hashlib.sha256(
                json.dumps(row["cells"]["PP61_A61"]["batch_indexes"]).encode()
            ).hexdigest(),
            "same_t61_batch_in_all_cells": len(
                {tuple(cell["batch_indexes"]) for cell in row["cells"].values()}
            )
            == 1,
            "cells": {
                name: {
                    key: value
                    for key, value in cell.items()
                    if key not in {"state_rows", "batch_indexes"}
                }
                for name, cell in row["cells"].items()
            },
            "factorial": row["factorial"],
        }
        for row in per_step
    ]
    finals = {
        "a0_factorial": factorial(
            {
                name: value["final"]["cumulative_cluster_specific_a0_movement"]
                for name, value in primary.items()
            }
        ),
        "margin_factorial": factorial(
            {
                name: value["final"]["final_cluster_mean_margin"]
                for name, value in primary.items()
            }
        ),
    }
    robustness_steps = []
    for step in STEPS:
        cells = {
            name: next(
                row for row in all_runs["T63"][name]["per_step"] if row["step"] == step
            )
            for name in CELL_SPECS
        }
        robustness_steps.append(
            {
                "step": step,
                "responses": {
                    name: cell["cluster_specific_a0_effect"]
                    for name, cell in cells.items()
                },
                "factorial": factorial(
                    {
                        name: cell["cluster_specific_a0_effect"]
                        for name, cell in cells.items()
                    }
                ),
            }
        )
    robustness = {
        "steps": robustness_steps,
        "final_a0_factorial": factorial(
            {
                name: value["final"]["cumulative_cluster_specific_a0_movement"]
                for name, value in all_runs["T63"].items()
            }
        ),
    }
    input_runs = {}
    for name, recipient, donor in (
        ("T61_with_T63_input", "T61", "T63"),
        ("T63_with_T61_input", "T63", "T61"),
    ):
        params = transplant_input_parameters(
            snapshots[recipient]["before"], snapshots[donor]["before"]
        )
        rows, model = run_sequence(
            params,
            snapshots[recipient]["optimizer"],
            lanes["T61"],
            batches["T61"],
            manifest["entries"],
        )
        input_runs[name] = final_response(
            params, model, manifest["entries"], lanes["T61"]["x"].shape[1]
        )
    adam_input_runs = {}
    for recipient, donor in (("T61", "T63"), ("T63", "T61")):
        optimizer = transplant_input_adam_state(
            snapshots[recipient]["optimizer"],
            snapshots[donor]["optimizer"],
            snapshots[recipient]["before"],
        )
        rows, model = run_sequence(
            snapshots[recipient]["before"],
            optimizer,
            lanes["T61"],
            batches["T61"],
            manifest["entries"],
        )
        adam_input_runs[f"{recipient}_input_adam_{donor}"] = final_response(
            snapshots[recipient]["before"],
            model,
            manifest["entries"],
            lanes["T61"]["x"].shape[1],
        )
    classification, next_experiment = classify(per_step, finals)
    g0_model = PolicyValueNet((96, 3), "residual_v3", lanes["T61"]["x"].shape[1])
    load_checkpoint_into_model(g0_model, paths["parent"])
    full_effect = (
        primary["P63_A61"]["final"]["final_cluster_mean_margin"]
        - primary["PP61_A61"]["final"]["final_cluster_mean_margin"]
    )
    input_effect = (
        input_runs["T61_with_T63_input"]["final_cluster_mean_margin"]
        - primary["PP61_A61"]["final"]["final_cluster_mean_margin"]
    )
    localization = {
        "geometry": input_geometry(
            snapshots["T61"]["before"],
            snapshots["T63"]["before"],
            tensor_snapshot(g0_model),
            snapshots["T61"]["optimizer"],
            snapshots["T63"]["optimizer"],
        ),
        "top_a0_units": unit_localization(
            snapshots["T61"]["before"],
            snapshots["T63"]["before"],
            manifest["entries"],
            lanes["T61"]["x"].shape[1],
        ),
        "input_parameter_transplants": input_runs,
        "input_layer_fraction": abs(input_effect) / max(abs(full_effect), 1e-12),
        "input_adam_transplants": adam_input_runs,
        "downstream_fixed_context": {
            "T61_downstream_T63_input": downstream_fixed_context(
                snapshots["T61"]["before"],
                snapshots["T63"]["before"],
                manifest["entries"],
                lanes["T61"]["x"].shape[1],
            ),
            "T63_downstream_T61_input": downstream_fixed_context(
                snapshots["T63"]["before"],
                snapshots["T61"]["before"],
                manifest["entries"],
                lanes["T61"]["x"].shape[1],
            ),
        },
    }
    result |= {
        "historical_checkpoint_sha256": {
            "T61": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
            "T63": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
        },
        "native_reproduction": native,
        "primary_t61_sequence": compact_per_step,
        "final_decomposition": finals,
        "t63_batch_robustness": robustness,
        "classification": classification,
        "next_experiment": next_experiment,
    }
    write_json(args.out_result, result)
    write_json(args.out_steps, {"steps": compact_per_step, "final": finals})
    write_json(args.out_input_localization, localization)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        "\n".join(
            [
                "# R61 Step-82 Parameter vs Adam Decomposition",
                "",
                "Inherited #321 classification: `earliest_divergence_mixed`.",
                "",
                "Step 82 was selected as the largest absolute replay x recipient interaction; this audit fixes replay and decomposes recipient state over steps 80-84.",
                "",
                "## Snapshot Verification",
                "",
                "```json",
                json.dumps(native, indent=2),
                "```",
                "",
                "## Five-Step Factorial Trajectories",
                "",
                "```json",
                json.dumps(compact_per_step, indent=2),
                "```",
                "",
                "## Cumulative Effects",
                "",
                "```json",
                json.dumps(finals, indent=2),
                "```",
                "",
                "## Input-Layer Parameter and Adam Localization",
                "",
                "```json",
                json.dumps(localization, indent=2),
                "```",
                "",
                "## T63-Batch Robustness",
                "",
                "```json",
                json.dumps(robustness, indent=2),
                "```",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Exactly one next experiment: {next_experiment}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
