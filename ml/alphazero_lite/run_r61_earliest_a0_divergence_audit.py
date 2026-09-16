#!/usr/bin/env python3
"""Dense causal audit of the first material R61 T61/T63 A0 divergence.

This is diagnostic-only.  It reads PR #320's SHA-verifiable per-step effects to
freeze one window, then reconstructs just the historical snapshots needed for
the registered one-step and sequence counterfactuals.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from collections import Counter
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
from ml.alphazero_lite.run_r61_activation_representation_audit import (
    patch_metrics,
    rescue,
    reverse_break,
    residual_v3_activations,
    legal_policy_metrics,
)
from ml.alphazero_lite.run_r61_early_trunk_replay_provenance_audit import (
    CLONE_TOLERANCE,
    EPOCHS,
    EXPECTED_SHA,
    FORMATION_END,
    GRAD_CLIP,
    LR,
    REPLAY,
    REPLAY_WEIGHTS,
    TRAINING_SEEDS,
    VALUE_WEIGHT,
    aggregate_effects,
    batch_hash,
    cohort,
    composition,
    evaluate_step,
    input_vector,
    model_from_snapshot,
    replay_metadata,
    tensor_snapshot,
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

SCHEMA = "azlite_r61_earliest_a0_divergence_audit_v1"
MATERIAL_FRACTION = 0.25
PERSISTENT_NEXT_STEPS = 5
PERSISTENT_REQUIRED = 4
REPLACEMENT_THRESHOLD = 0.01


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def effect_curve(
    t61: list[dict[str, Any]], t63: list[dict[str, Any]]
) -> list[dict[str, float | int]]:
    """Reconstruct cumulative effects by matched optimizer step, never rounded totals."""
    by_step = [{int(row["optimizer_step"]): row for row in lane} for lane in (t61, t63)]
    rows, c61, c63 = [], 0.0, 0.0
    for step in range(1, FORMATION_END + 1):
        if step not in by_step[0] or step not in by_step[1]:
            raise RuntimeError("early_window_trace_missing_step")
        c61 += float(by_step[0][step]["cluster_specific_a0_effect"])
        c63 += float(by_step[1][step]["cluster_specific_a0_effect"])
        rows.append({"step": step, "c61": c61, "c63": c63, "difference": c61 - c63})
    return rows


def first_material_divergence(curve: list[dict[str, float | int]]) -> tuple[int, float]:
    final = float(curve[-1]["difference"])
    threshold = MATERIAL_FRACTION * abs(final)
    if final == 0:
        raise RuntimeError("early_window_zero_final_difference")
    sign = 1 if final > 0 else -1
    for index, row in enumerate(curve):
        value = float(row["difference"])
        following = curve[index + 1 : index + 1 + PERSISTENT_NEXT_STEPS]
        persistent = sum(
            (1 if float(candidate["difference"]) > 0 else -1) == sign
            and abs(float(candidate["difference"])) >= threshold
            for candidate in following
        )
        if (
            (1 if value > 0 else -1) == sign
            and abs(value) >= threshold
            and persistent >= PERSISTENT_REQUIRED
        ):
            return int(row["step"]), threshold
    raise RuntimeError("early_window_material_divergence_not_found")


def frozen_window(step: int) -> list[int]:
    return list(range(max(1, step - 4), min(FORMATION_END, step + 4) + 1))


def max_tensor_error(model: PolicyValueNet, expected: dict[str, torch.Tensor]) -> float:
    return max(
        float((parameter.detach().cpu() - expected[name]).abs().max())
        for name, parameter in model.named_parameters()
    )


def adam_update(
    before: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    batch_x: np.ndarray,
    batch_p: np.ndarray,
    batch_v: np.ndarray,
) -> tuple[PolicyValueNet, torch.optim.Adam]:
    """The unmodified historical policy/value, clipping, and Adam update."""
    model = model_from_snapshot(before, batch_x.shape[1]).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    x, target_p, target_v = map(torch.from_numpy, (batch_x, batch_p, batch_v))
    mask = torch.from_numpy(legal_mask_matrix_for_encoded_states(batch_x))
    logits, prediction = model(x)
    loss = compute_policy_cross_entropy(
        logits.masked_fill(mask <= 0, -1e9), target_p
    ).mean()
    loss += (
        VALUE_WEIGHT
        * compute_value_loss_vector(
            prediction, target_v, value_loss="huber", huber_delta=1.0
        ).mean()
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return model.eval(), optimizer


def reconstruct_lane(
    paths: dict[str, Any],
    manifest: dict[str, Any],
    seed: str,
    steps: set[int],
    workdir: Path,
) -> dict[str, Any]:
    """Replay unchanged R61 training while retaining only selected state snapshots."""
    set_seed(TRAINING_SEEDS[seed])
    inputs = [paths["replays"][REPLAY], *paths["fixed"]]
    x, p, v, indexes = load_jsonl_replay(
        inputs,
        list(REPLAY_WEIGHTS),
        policy_target_mode="sharpened",
        value_target_mode="sharpened",
    )
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, paths["parent"])
    snapshots: dict[int, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}

    def callback(phase: str, context: dict[str, Any]) -> None:
        step = len(trace) + (1 if phase == "before" else 0)
        if phase == "before":
            if step in steps:
                pending["before"] = tensor_snapshot(model)
                pending["optimizer"] = copy.deepcopy(context["optimizer"].state_dict())
            pending["indexes"] = [int(value) for value in context["batch_indexes"]]
            return
        current_indexes = pending.pop("indexes")
        trace.append(
            {
                "optimizer_step": len(trace) + 1,
                "batch_indexes": current_indexes,
                "batch_hash": batch_hash(current_indexes),
            }
        )
        if len(trace) in steps:
            snapshots[len(trace)] = {
                "before": pending.pop("before"),
                "after": tensor_snapshot(model),
                "optimizer": pending.pop("optimizer"),
            }

    train(
        model,
        x,
        p,
        v,
        indexes,
        epochs=EPOCHS,
        batch_size=512,
        lr=LR,
        device=torch.device("cpu"),
        value_loss_weight=VALUE_WEIGHT,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.1,
        grad_clip=GRAD_CLIP,
        save_top_k=3,
        lr_scheduler="none",
        step_callback=callback,
    )
    checkpoint = workdir / "reconstructed" / f"{seed}.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint, **checkpoint_from_model(model))
    return {
        "sha": sha256_file(checkpoint),
        "snapshots": snapshots,
        "trace": trace,
        "x": x,
        "p": p,
        "v": v,
        "metadata": replay_metadata(inputs, p, v),
        "manifest": manifest,
    }


def anchor_effect(rows: list[dict[str, Any]]) -> float:
    return statistics.fmean(
        float(row["a0_only_margin_delta"]) for row in rows if row["cohort"] == "anchor"
    )


def cell(
    before: dict[str, torch.Tensor],
    optimizer: dict[str, Any],
    donor: dict[str, Any],
    indexes: list[int],
    entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], PolicyValueNet, torch.optim.Adam]:
    after, updated_optimizer = adam_update(
        before, optimizer, donor["x"][indexes], donor["p"][indexes], donor["v"][indexes]
    )
    before_model = model_from_snapshot(before, donor["x"].shape[1])
    rows, effects = evaluate_step(before_model, after, entries)
    effects |= {
        "anchor_a0_only_margin_effect": anchor_effect(rows),
        "input_layer_update_norm": float(
            torch.linalg.vector_norm(
                input_vector(tensor_snapshot(after)) - input_vector(before)
            )
        ),
        "post_context": {
            key: value
            for key, value in aggregate_effects(
                [
                    {
                        **row,
                        "a0_only_margin_delta": row[
                            "a0_only_margin_delta_post_context"
                        ],
                    }
                    for row in rows
                ]
            ).items()
        },
        "state_rows": rows,
    }
    return effects, after, updated_optimizer


def factorial(cells: dict[str, dict[str, Any]]) -> dict[str, float]:
    a, b, c, d = (
        float(cells[name]["cluster_specific_a0_effect"])
        for name in ("A", "B", "C", "D")
    )
    return {
        "batch_effect": 0.5 * ((b - a) + (d - c)),
        "state_effect": 0.5 * ((c - a) + (d - b)),
        "interaction": (d - c) - (b - a),
    }


def classify_step(cells: dict[str, dict[str, Any]]) -> str:
    a, b, c = (
        float(cells[name]["cluster_specific_a0_effect"]) for name in ("A", "B", "C")
    )
    # Harm/protection are defined on the pre-registered primary response.
    if a < 0 and c >= 0:
        return "state_dependent_harm"
    if a < 0 and c < 0:
        return "content_stable_harm"
    if b - a >= REPLACEMENT_THRESHOLD:
        return "protective_t63_batch"
    if a < 0 and b < 0:
        return "both_batches_harmful"
    if a > 0 and b > 0:
        return "both_batches_protective"
    return "mixed_other"


def overlap(
    left: list[int], right: list[int], metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    left_set, right_set = set(left), set(right)
    state_left = {metadata[index]["canonical_state_hash"] for index in left_set}
    state_right = {metadata[index]["canonical_state_hash"] for index in right_set}
    shared_states = state_left & state_right
    policy_tv = []
    for key in shared_states:
        lrow = next(
            metadata[index]
            for index in left_set
            if metadata[index]["canonical_state_hash"] == key
        )
        rrow = next(
            metadata[index]
            for index in right_set
            if metadata[index]["canonical_state_hash"] == key
        )
        policy_tv.append(
            0.5
            * sum(
                abs(a - b)
                for a, b in zip(
                    lrow["policy_target"], rrow["policy_target"], strict=True
                )
            )
        )
    left_sources, right_sources = (
        Counter(metadata[index]["source"] for index in left),
        Counter(metadata[index]["source"] for index in right),
    )
    return {
        "compact_row_jaccard": len(left_set & right_set)
        / max(len(left_set | right_set), 1),
        "canonical_state_jaccard": len(shared_states)
        / max(len(state_left | state_right), 1),
        "same_source_fraction": sum((left_sources & right_sources).values())
        / max(len(left), len(right), 1),
        "shared_state_policy_target_tv": statistics.fmean(policy_tv)
        if policy_tv
        else None,
    }


def batch_provenance(
    indexes: list[int], metadata: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Persist the immutable replay identity and targets for an actual batch."""
    fields = (
        "compact_index",
        "source",
        "source_artifact",
        "source_sha256",
        "local_jsonl_row",
        "canonical_state_hash",
        "policy_target",
        "value_target",
    )
    return [{field: metadata[index][field] for field in fields} for index in indexes]


def sequence(
    lane: dict[str, Any],
    recipient_step: int,
    donor: dict[str, Any],
    donor_steps: list[int],
    entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], PolicyValueNet]:
    snapshot = lane["snapshots"][recipient_step]
    before = snapshot["before"]
    model = model_from_snapshot(before, lane["x"].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    for step in donor_steps:
        indexes = donor["trace"][step - 1]["batch_indexes"]
        model, optimizer = adam_update(
            tensor_snapshot(model),
            optimizer.state_dict(),
            donor["x"][indexes],
            donor["p"][indexes],
            donor["v"][indexes],
        )
    rows, effects = evaluate_step(
        model_from_snapshot(before, lane["x"].shape[1]), model, entries
    )
    cluster_margins = []
    anchor_margin = None
    with torch.no_grad():
        for entry in entries:
            metrics = legal_policy_metrics(
                residual_v3_activations(
                    model,
                    torch.tensor(
                        [encode_state(entry["state"], input_encoding="kalah_v3")],
                        dtype=torch.float32,
                    ),
                )["A5"],
                entry,
            )
            if cohort(entry) == "cluster":
                cluster_margins.append(float(metrics["margin"]))
            if entry["id"] == ANCHOR_ID:
                anchor_margin = float(metrics["margin"])
    cluster_movements = [
        float(row["movement"]["l2"]) for row in rows if row["cohort"] == "cluster"
    ]
    return effects | {
        "steps_applied": len(donor_steps),
        "cluster_mean_margin": statistics.fmean(cluster_margins),
        "anchor_margin": anchor_margin,
        "cluster_a0_representation_movement": statistics.fmean(cluster_movements),
        "state_rows": rows,
    }, model


def patching(
    t61: PolicyValueNet, t63: PolicyValueNet, manifest: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    for entry in manifest["entries"]:
        x = torch.tensor(
            [encode_state(entry["state"], input_encoding="kalah_v3")],
            dtype=torch.float32,
        )
        native61 = legal_policy_metrics(residual_v3_activations(t61, x)["A5"], entry)
        native63 = legal_policy_metrics(residual_v3_activations(t63, x)["A5"], entry)
        forward, reverse = (
            patch_metrics(t61, t63, "A0", x, entry),
            patch_metrics(t63, t61, "A0", x, entry),
        )
        rows.append(
            {
                "id": entry["id"],
                "cohort": cohort(entry),
                "forward": forward,
                "reverse": reverse,
                "rescue": rescue(native61, forward),
                "reverse_break": reverse_break(native63, reverse),
            }
        )
    cluster = [row for row in rows if row["cohort"] == "cluster"]
    return {
        "rows": rows,
        "cluster_rescue_fraction": sum(row["rescue"] for row in cluster)
        / max(len(cluster), 1),
        "cluster_reverse_break_fraction": sum(row["reverse_break"] for row in cluster)
        / max(len(cluster), 1),
    }


def order_test_eligible(
    window_rows: list[dict[str, Any]], metadata: list[dict[str, Any]]
) -> bool:
    left = {
        metadata[index]["canonical_state_hash"]
        for row in window_rows
        for index in row["t61_batch"]["batch_indexes"]
    }
    right = {
        metadata[index]["canonical_state_hash"]
        for row in window_rows
        for index in row["t63_batch"]["batch_indexes"]
    }
    return len(left & right) / max(len(left | right), 1) >= 0.50


def hard_classification(
    rows: list[dict[str, Any]],
    sequences: dict[str, dict[str, Any]],
    hybrid: dict[str, Any],
) -> tuple[str, str]:
    labels = Counter(row["classification"] for row in rows)
    interaction = statistics.fmean(
        abs(float(row["factorial"]["interaction"])) for row in rows
    )
    state = statistics.fmean(
        abs(float(row["factorial"]["state_effect"])) for row in rows
    )
    batch = statistics.fmean(
        abs(float(row["factorial"]["batch_effect"])) for row in rows
    )
    gain = (
        sequences["S2"]["cluster_mean_margin"] - sequences["S1"]["cluster_mean_margin"]
    )
    damage = (
        sequences["S4"]["cluster_mean_margin"] - sequences["S3"]["cluster_mean_margin"]
    )
    if hybrid.get("adam_state_primary"):
        return (
            "earliest_divergence_adam_state_primary",
            "trace Adam first/second moments on input_layer over the few steps immediately preceding this divergence.",
        )
    if (
        gain >= 0.05
        and damage <= -0.05
        and labels["content_stable_harm"] + labels["protective_t63_batch"]
        > len(rows) / 2
        and max(state, interaction) <= batch
    ):
        return (
            "earliest_divergence_replay_sequence_causal",
            "run one historical-sequence order/content ablation on frozen R61 that changes only this identified early-window sequence while preserving total examples.",
        )
    if (
        labels["state_dependent_harm"]
        > max(
            1,
            sum(row["cells"]["A"]["cluster_specific_a0_effect"] < 0 for row in rows)
            / 2,
        )
        and gain < 0.05
        and max(state, interaction) > batch
    ):
        if hybrid.get("parameter_representation_primary"):
            return (
                "earliest_divergence_parameter_representation_primary",
                "trace which input-layer parameter directions / A0 units create the cluster-specific divergence immediately before t*.",
            )
        return (
            "earliest_divergence_parameter_state_causal",
            "audit the parameter-vs-Adam contribution at the first state-dependent divergence using a short local trajectory.",
        )
    if abs(gain) < 0.05 and abs(damage) < 0.05:
        return (
            "earliest_divergence_not_localized",
            "test one G0-A0 feature-retention diagnostic on ordinary replay states, without using the forensic cluster for training.",
        )
    return (
        "earliest_divergence_mixed",
        "take the single earliest window step with the largest absolute replay×state interaction and perform parameter-vs-Adam recipient decomposition there only.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior-trace",
        type=Path,
        default=ROOT / ".tmp/r61-early-trunk-provenance/per-step-provenance.json",
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-result", type=Path, required=True)
    parser.add_argument("--out-window", type=Path, required=True)
    parser.add_argument("--out-counterfactuals", type=Path, required=True)
    parser.add_argument("--out-sequences", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    prior = json.loads(args.prior_trace.read_text(encoding="utf-8"))
    curve = effect_curve(prior["per_step"]["T61"], prior["per_step"]["T63"])
    t_star, threshold = first_material_divergence(curve)
    window = frozen_window(t_star)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "inherited_classification": "early_trunk_provenance_heterogeneous",
        "g0_sha256": G0_SHA,
        "curve": curve,
        "d_final": curve[-1]["difference"],
        "material_threshold": threshold,
        "first_material_divergence_step": t_star,
        "earliest_divergence_window": window,
        "guardrails": {
            "replay_mutation": False,
            "self_play": False,
            "training_intervention": False,
            "exact_labels_as_training_targets": False,
            "promotion": False,
            "frozen_cluster_training_ineligible": True,
        },
    }
    write_json(
        args.out_window,
        {
            key: result[key]
            for key in (
                "schema",
                "d_final",
                "material_threshold",
                "first_material_divergence_step",
                "earliest_divergence_window",
            )
        },
    )
    if not args.execute:
        result |= {"classification": "planned", "next_experiment": "not run"}
        write_json(args.out_result, result)
        return 0
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
    lanes = {
        seed: reconstruct_lane(paths, manifest, seed, set(window), args.workdir)
        for seed in TRAINING_SEEDS
    }
    baseline = {
        seed: {
            "expected_sha256": EXPECTED_SHA[seed],
            "reconstructed_sha256": lanes[seed]["sha"],
            "reproduced": lanes[seed]["sha"] == EXPECTED_SHA[seed],
        }
        for seed in TRAINING_SEEDS
    }
    if not all(row["reproduced"] for row in baseline.values()):
        raise RuntimeError("early_window_counterfactual_invalid")
    dense = []
    for step in window:
        cells = {}
        for name, recipient, donor in (
            ("A", "T61", "T61"),
            ("B", "T61", "T63"),
            ("C", "T63", "T61"),
            ("D", "T63", "T63"),
        ):
            snap, indexes = (
                lanes[recipient]["snapshots"][step],
                lanes[donor]["trace"][step - 1]["batch_indexes"],
            )
            value, after, _ = cell(
                snap["before"],
                snap["optimizer"],
                lanes[donor],
                indexes,
                manifest["entries"],
            )
            if name in {"A", "D"}:
                value["historical_max_abs_tensor_error"] = max_tensor_error(
                    after, snap["after"]
                )
            cells[name] = value
        if any(
            float(cells[name].get("historical_max_abs_tensor_error", 0))
            > CLONE_TOLERANCE
            for name in ("A", "D")
        ):
            raise RuntimeError("early_window_counterfactual_invalid")
        dense.append(
            {
                "step": step,
                "t61_batch": lanes["T61"]["trace"][step - 1]
                | {
                    "rows": batch_provenance(
                        lanes["T61"]["trace"][step - 1]["batch_indexes"],
                        lanes["T61"]["metadata"],
                    )
                },
                "t63_batch": lanes["T63"]["trace"][step - 1]
                | {
                    "rows": batch_provenance(
                        lanes["T63"]["trace"][step - 1]["batch_indexes"],
                        lanes["T63"]["metadata"],
                    )
                },
                "cells": cells,
                "factorial": factorial(cells),
                "classification": classify_step(cells),
            }
        )
    start = window[0]
    sequences, models = {}, {}
    for name, recipient, donor in (
        ("S1", "T61", "T61"),
        ("S2", "T61", "T63"),
        ("S3", "T63", "T63"),
        ("S4", "T63", "T61"),
    ):
        sequences[name], models[name] = sequence(
            lanes[recipient], start, lanes[donor], window, manifest["entries"]
        )
    for name, lane in (("S1", "T61"), ("S3", "T63")):
        sequences[name]["historical_max_abs_tensor_error"] = max_tensor_error(
            models[name], lanes[lane]["snapshots"][window[-1]]["after"]
        )
    if any(
        sequences[name]["historical_max_abs_tensor_error"] > CLONE_TOLERANCE
        for name in ("S1", "S3")
    ):
        raise RuntimeError("early_window_counterfactual_invalid")
    overlaps = [
        {
            "step": row["step"],
            **overlap(
                row["t61_batch"]["batch_indexes"],
                row["t63_batch"]["batch_indexes"],
                lanes["T61"]["metadata"],
            ),
        }
        for row in dense
    ]
    max_row = max(
        dense,
        key=lambda row: (
            abs(float(row["factorial"]["interaction"])),
            -int(row["step"]),
        ),
    )
    step, t61_snap, t63_snap = (
        max_row["step"],
        lanes["T61"]["snapshots"][max_row["step"]],
        lanes["T63"]["snapshots"][max_row["step"]],
    )
    # Adam state is compatible only because both historical models share the exact full scope.
    hybrid = {}
    for name, parameters, moments in (
        ("T61_parameters_T63_adam", t61_snap, t63_snap),
        ("T63_parameters_T61_adam", t63_snap, t61_snap),
    ):
        value, _, _ = cell(
            parameters["before"],
            moments["optimizer"],
            lanes["T61"],
            lanes["T61"]["trace"][step - 1]["batch_indexes"],
            manifest["entries"],
        )
        hybrid[name] = value
    a = max_row["cells"]["A"]["cluster_specific_a0_effect"]
    adam_delta = float(
        hybrid["T61_parameters_T63_adam"]["cluster_specific_a0_effect"]
    ) - float(a)
    parameter_delta = float(
        hybrid["T63_parameters_T61_adam"]["cluster_specific_a0_effect"]
    ) - float(a)
    hybrid |= {
        "adam_state_delta": adam_delta,
        "parameter_delta": parameter_delta,
        "adam_state_primary": abs(adam_delta) >= REPLACEMENT_THRESHOLD
        and abs(adam_delta) > abs(parameter_delta),
        "parameter_representation_primary": abs(parameter_delta)
        >= REPLACEMENT_THRESHOLD
        and abs(parameter_delta) > abs(adam_delta),
    }
    classification, next_experiment = hard_classification(dense, sequences, hybrid)
    order_eligible = order_test_eligible(dense, lanes["T61"]["metadata"])
    result |= {
        "artifacts": artifacts,
        "baseline_reproduction": baseline,
        "window_context": {
            seed: [prior["per_step"][seed][step - 1] for step in window]
            for seed in TRAINING_SEEDS
        },
        "batch_composition": {
            seed: composition(
                [lanes[seed]["trace"][step - 1] for step in window],
                lanes[seed]["metadata"],
            )
            for seed in TRAINING_SEEDS
        },
        "row_overlap": overlaps,
        "order_micro_test": {
            "eligible": order_eligible,
            "status": "not_implemented_when_eligible"
            if order_eligible
            else "skipped_below_50_percent_canonical_overlap",
        },
        "max_interaction_step": step,
        "hybrid_recipient": hybrid,
        "sequence_patching": {
            "historical_sequences": patching(models["S1"], models["S3"], manifest),
            "replacement_sequences": patching(models["S2"], models["S4"], manifest),
        },
        "classification": classification,
        "next_experiment": next_experiment,
    }
    write_json(args.out_result, result)
    write_json(
        args.out_counterfactuals,
        {"cells": dense, "historical_reproduction_valid": True},
    )
    write_json(
        args.out_sequences,
        {
            "sequences": sequences,
            "t61_sequence_replacement_gain": {
                "cluster_mean_margin": sequences["S2"]["cluster_mean_margin"]
                - sequences["S1"]["cluster_mean_margin"],
                "anchor_margin": sequences["S2"]["anchor_margin"]
                - sequences["S1"]["anchor_margin"],
            },
            "t63_sequence_damage": {
                "cluster_mean_margin": sequences["S4"]["cluster_mean_margin"]
                - sequences["S3"]["cluster_mean_margin"],
                "anchor_margin": sequences["S4"]["anchor_margin"]
                - sequences["S3"]["anchor_margin"],
            },
        },
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    compact_cells = [
        {
            "step": row["step"],
            "responses": {
                name: row["cells"][name]["cluster_specific_a0_effect"]
                for name in "ABCD"
            },
            "factorial": row["factorial"],
            "classification": row["classification"],
        }
        for row in dense
    ]
    compact_sequences = {
        name: {
            key: value
            for key, value in sequence_row.items()
            if key not in {"state_rows"}
        }
        for name, sequence_row in sequences.items()
    }
    args.out_report.write_text(
        "\n".join(
            [
                "# R61 Earliest A0 Divergence Audit",
                "",
                "Inherited #320 classification: `early_trunk_provenance_heterogeneous`.",
                "",
                "## Difference Curve",
                "",
                "The following mechanically reconstructs `C61`, `C63`, and `D` through step 108 from the #320 trace.",
                "",
                "```json",
                json.dumps(curve, indent=2),
                "```",
                "",
                "## Deterministic Window",
                "",
                f"`D(108)={result['d_final']}`, threshold `0.25 * abs(D108) = {threshold}`, first persistent material step `{t_star}`, window `{window}`.",
                "",
                "## Window Trajectories",
                "",
                "Per-step cluster/control/specific A0 effects, cumulative values, anchor effect/margin, support flips, and input update norm are retained without aggregation in `window_context`.",
                "",
                "```json",
                json.dumps(result["window_context"], indent=2),
                "```",
                "",
                "## Dense 2x2 And Factorial Decomposition",
                "",
                "```json",
                json.dumps(compact_cells, indent=2),
                "```",
                "",
                "## Sequences",
                "",
                "```json",
                json.dumps(compact_sequences, indent=2),
                "```",
                "",
                "## Batch Context And Overlap",
                "",
                "```json",
                json.dumps(
                    {
                        "composition": result["batch_composition"],
                        "overlap": overlaps,
                        "order_micro_test": result["order_micro_test"],
                    },
                    indent=2,
                ),
                "```",
                "",
                "## Hybrid And A0 Causality",
                "",
                "```json",
                json.dumps(
                    {
                        "max_interaction_step": step,
                        "hybrid": hybrid,
                        "patching": result["sequence_patching"],
                    },
                    indent=2,
                ),
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
