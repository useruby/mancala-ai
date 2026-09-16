#!/usr/bin/env python3
"""Observational provenance audit for R61's A0 representation split.

This runner deliberately uses ``train`` for both historical trajectories.  Its
callbacks copy observations only; replay, losses, optimizer semantics, and
trainable scope are the original R61 configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import decode_state
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_table
from ml.alphazero_lite.run_g1_replay_optimizer_crossover import (
    FROZEN_SET_SHA,
    G0_SHA,
    artifact_paths,
    sha256_file,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import (
    MANIFEST_SCHEMA,
    approximate_phase,
    state_sha,
)
from ml.alphazero_lite.run_internal_state_cluster_provenance_audit import (
    cluster_signature,
)
from ml.alphazero_lite.run_r61_activation_representation_audit import (
    continue_policy_from_stage,
    legal_policy_metrics,
    residual_v3_activations,
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
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    train,
)

SCHEMA = "azlite_r61_early_trunk_replay_provenance_audit_v1"
TRAINING_SEEDS = {"T61": 61, "T63": 63}
EXPECTED_SHA = {
    "T61": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "T63": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
}
EPOCHS, BATCH_SIZE, LR, GRAD_CLIP, VALUE_WEIGHT = 4, 512, 0.001, 1.0, 0.3
FORMATION_END = 108
EPS = 1e-12
CLONE_TOLERANCE = 2e-6
FAMILY_FIELDS = (
    "source",
    "source_phase",
    "source_player",
    "source_legal_count",
    "source_capture_available",
    "source_extra_turn_available",
    "source_value_sign",
    "source_structural_neighbor",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def tensor_snapshot(model: PolicyValueNet) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }


def model_from_snapshot(
    before: dict[str, torch.Tensor], input_size: int
) -> PolicyValueNet:
    # PolicyValueNet initialization consumes the global torch RNG even though
    # every tensor is overwritten below. Restore it so callbacks cannot alter
    # future epoch permutations in the historical training trajectory.
    rng_state = torch.get_rng_state()
    width = int(before["input_layer.weight"].shape[0])
    block_count = len(
        {name.split(".")[1] for name in before if name.startswith("residual_layers.")}
    )
    model = PolicyValueNet((width, block_count), "residual_v3", input_size)
    torch.set_rng_state(rng_state)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(before[name])
    return model.eval()


def input_vector(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            values["input_layer.weight"].reshape(-1).float(),
            values["input_layer.bias"].reshape(-1).float(),
        ]
    )


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    )
    return None if denominator == 0 else float(torch.dot(left, right) / denominator)


def batch_hash(indexes: list[int]) -> str:
    """Hash ordered compact indexes; repeats are meaningful replay exposures."""
    return hashlib.sha256(
        json.dumps([int(value) for value in indexes], separators=(",", ":")).encode()
    ).hexdigest()


def a0_movement(before: torch.Tensor, after: torch.Tensor) -> dict[str, float | int]:
    delta = after - before
    before_active, after_active = before > 0, after > 0
    return {
        "l2": float(torch.linalg.vector_norm(delta)),
        "normalized_l2": float(
            torch.linalg.vector_norm(delta)
            / max(float(torch.linalg.vector_norm(before)), EPS)
        ),
        "cosine": cosine(before.reshape(-1), after.reshape(-1)) or 0.0,
        "support_flips": int((before_active != after_active).sum()),
        "activated": int((~before_active & after_active).sum()),
        "deactivated": int((before_active & ~after_active).sum()),
    }


def cohort(entry: dict[str, Any]) -> str:
    membership = entry["membership"]
    return (
        "anchor"
        if membership == "cluster_anchor"
        else "controls"
        if membership == "matched_control"
        else "cluster"
    )


def margin_from_a0(
    model: PolicyValueNet, a0: torch.Tensor, entry: dict[str, Any]
) -> float:
    return float(
        legal_policy_metrics(continue_policy_from_stage(model, "A0", a0), entry)[
            "margin"
        ]
    )


def state_probe_loss(
    model: PolicyValueNet, entries: list[dict[str, Any]]
) -> torch.Tensor:
    losses = []
    for entry in entries:
        x = torch.tensor(
            [encode_state(entry["state"], input_encoding="kalah_v3")],
            dtype=torch.float32,
        )
        logits, _ = model(x)
        legal = entry["legal_actions"]
        probability = torch.softmax(logits[0, legal], 0)
        selected = torch.tensor(
            [action in entry["exact_outcome_optimal_actions"] for action in legal]
        )
        losses.append(-torch.log(probability[selected].sum()))
    return torch.stack(losses).mean()


def probe_gradient(
    model: PolicyValueNet, entries: list[dict[str, Any]]
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        state_probe_loss(model, entries),
        [model.input_layer.weight, model.input_layer.bias],
    )
    return torch.cat([value.detach().reshape(-1).float().cpu() for value in gradients])


def replay_metadata(
    paths: list[Path], compact_p: np.ndarray, compact_v: np.ndarray
) -> list[dict[str, Any]]:
    """Map compact loader order to immutable source rows, never state inference."""
    provenance = json.loads(
        (
            ROOT
            / "docs/data/alphazero-lite-internal-state-cluster-provenance-audit.json"
        ).read_text()
    )
    signature = provenance["selected_cluster"]["definition"]
    exact = set(provenance["selected_cluster"]["states"])
    rows: list[dict[str, Any]] = []
    index = 0
    for source_order, path in enumerate(paths):
        source_sha = sha256_file(path)
        source_identity = "dynamic" if source_order == 0 else f"fixed:{path.stem}"
        with path.open(encoding="utf-8") as handle:
            for local_row, line in enumerate(handle, 1):
                raw = json.loads(line)
                state = decode_state(raw["state"])
                legal = KalahGame.from_state(state).possible_moves()
                consequences = [
                    row for row in move_consequence_table(state) if row["legal"]
                ]
                target = compact_p[index]
                entropy = -float(np.sum(target * np.log(np.clip(target, EPS, 1.0))))
                key = canonical_state_key(state)
                value = float(compact_v[index, 0])
                legal_count = len(legal)
                rows.append(
                    {
                        "compact_index": index,
                        "source": source_identity,
                        "source_artifact": str(path),
                        "source_sha256": source_sha,
                        "local_jsonl_row": local_row,
                        "canonical_state_hash": state_sha(state),
                        "current_player": state["current_player"],
                        "legal_action_count": legal_count,
                        "phase": approximate_phase(state),
                        "capture_available": any(
                            item["produces_capture"] for item in consequences
                        ),
                        "extra_turn_available": any(
                            item["gives_extra_turn"] for item in consequences
                        ),
                        "policy_target": target.tolist(),
                        "policy_target_entropy": entropy,
                        "policy_target_max_probability": float(target.max()),
                        "policy_target_top_action": int(target.argmax()),
                        "value_target": value,
                        "value_target_sign": "positive"
                        if value > 0
                        else "negative"
                        if value < 0
                        else "zero",
                        "structural_neighbor": key not in exact
                        and cluster_signature(state) == signature,
                        "exact_structural_cluster": key in exact,
                        "effective_replay_weight": int(REPLAY_WEIGHTS[source_order]),
                    }
                )
                index += 1
    if index != len(compact_p):
        raise RuntimeError("compact_replay_provenance_length_mismatch")
    return rows


def aggregate_effects(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "cluster_a0_effect": 0.0,
            "control_a0_effect": 0.0,
            "cluster_specific_a0_effect": 0.0,
        }
    by_cohort: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cohort[row["cohort"]].append(float(row["a0_only_margin_delta"]))
    cluster = by_cohort["cluster"]
    controls = by_cohort["controls"]
    cluster_effect = statistics.fmean(cluster) if cluster else 0.0
    control_effect = statistics.fmean(controls) if controls else 0.0
    return {
        "cluster_a0_effect": cluster_effect,
        "control_a0_effect": control_effect,
        "cluster_specific_a0_effect": cluster_effect - control_effect,
    }


def ranked_steps(rows: list[dict[str, Any]]) -> dict[str, list[int] | dict[str, float]]:
    ordered = sorted(
        rows, key=lambda row: (row["cluster_specific_a0_effect"], row["optimizer_step"])
    )
    negative = [
        -float(row["cluster_specific_a0_effect"])
        for row in ordered
        if row["cluster_specific_a0_effect"] < 0
    ]
    total = sum(negative)
    return {
        "worst_1": [row["optimizer_step"] for row in ordered[:1]],
        "worst_5": [row["optimizer_step"] for row in ordered[:5]],
        "worst_10": [row["optimizer_step"] for row in ordered[:10]],
        "worst_20": [row["optimizer_step"] for row in ordered[:20]],
        "best_20": [row["optimizer_step"] for row in ordered[-20:][::-1]],
        "negative_concentration": {
            str(count): sum(negative[:count]) / total if total else 0.0
            for count in (1, 5, 10, 20)
        },
    }


def family_id(row: dict[str, Any], field: str) -> str:
    values = {
        "source": row["source"],
        "source_phase": f"{row['source']}|{row['phase']}",
        "source_player": f"{row['source']}|p{row['current_player']}",
        "source_legal_count": f"{row['source']}|legal{row['legal_action_count']}",
        "source_capture_available": f"{row['source']}|capture={row['capture_available']}",
        "source_extra_turn_available": f"{row['source']}|extra={row['extra_turn_available']}",
        "source_value_sign": f"{row['source']}|value={row['value_target_sign']}",
        "source_structural_neighbor": f"{row['source']}|neighbor={row['structural_neighbor']}",
    }
    return f"{field}:{values[field]}"


def select_candidate_family(families: list[dict[str, Any]]) -> dict[str, Any] | None:
    qualified = [
        row
        for row in families
        if row["harmful_batches"] >= 5
        and row["enrichment"] >= 2
        and row["negative_per_exposure"] < row["baseline_negative_per_exposure"]
        and row["median_cluster_probe_alignment"] < 0
        and row["median_control_probe_alignment"]
        > row["median_cluster_probe_alignment"]
    ]
    return (
        sorted(
            qualified,
            key=lambda row: (
                -row["negative_contribution"],
                -row["enrichment"],
                row["family_id"],
            ),
        )[0]
        if qualified
        else None
    )


def clone_adam_step(
    before: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    batch_x: np.ndarray,
    batch_p: np.ndarray,
    batch_v: np.ndarray,
) -> PolicyValueNet:
    model = model_from_snapshot(before, batch_x.shape[1]).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    x, target_p, target_v = (
        torch.from_numpy(batch_x),
        torch.from_numpy(batch_p),
        torch.from_numpy(batch_v),
    )
    mask = torch.from_numpy(legal_mask_matrix_for_encoded_states(batch_x))
    logits, prediction = model(x)
    loss = (
        compute_policy_cross_entropy(
            logits.masked_fill(mask <= 0, -1e9), target_p
        ).mean()
        + VALUE_WEIGHT
        * compute_value_loss_vector(
            prediction, target_v, value_loss="huber", huber_delta=1.0
        ).mean()
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return model.eval()


def evaluate_step(
    before_model: PolicyValueNet,
    after_model: PolicyValueNet,
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    with torch.no_grad():
        for entry in entries:
            x = torch.tensor(
                [encode_state(entry["state"], input_encoding="kalah_v3")],
                dtype=torch.float32,
            )
            pre_a0, post_a0 = (
                residual_v3_activations(before_model, x)["A0"],
                residual_v3_activations(after_model, x)["A0"],
            )
            pre_margin = margin_from_a0(before_model, pre_a0, entry)
            rows.append(
                {
                    "id": entry["id"],
                    "cohort": cohort(entry),
                    "a0_before": pre_a0.reshape(-1).tolist(),
                    "a0_after": post_a0.reshape(-1).tolist(),
                    "movement": a0_movement(pre_a0, post_a0),
                    "a0_only_margin_delta": margin_from_a0(before_model, post_a0, entry)
                    - pre_margin,
                    "a0_only_margin_delta_post_context": margin_from_a0(
                        after_model, post_a0, entry
                    )
                    - margin_from_a0(after_model, pre_a0, entry),
                }
            )
    return rows, aggregate_effects(rows)


def run_lane(
    paths: dict[str, Any],
    manifest: dict[str, Any],
    seed_name: str,
    workdir: Path,
    instrumented: bool,
) -> dict[str, Any]:
    set_seed(TRAINING_SEEDS[seed_name])
    inputs = [paths["replays"][REPLAY], *paths["fixed"]]
    x, p, v, indexes = load_jsonl_replay(
        inputs,
        list(REPLAY_WEIGHTS),
        policy_target_mode="sharpened",
        value_target_mode="sharpened",
    )
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, paths["parent"])
    traces: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, Any]] = {}
    pending: dict[str, Any] = {}
    entries = manifest["entries"]

    def callback(phase: str, context: dict[str, Any]) -> None:
        if phase == "before":
            pending["step"] = len(traces) + 1
            pending["before"] = tensor_snapshot(model)
            pending["optimizer"] = copy.deepcopy(context["optimizer"].state_dict())
            if instrumented:
                raw = input_vector(context["raw_gradients"])
                was_training = model.training
                model.eval()
                pending["probe"] = {
                    "cluster": probe_gradient(
                        model,
                        [entry for entry in entries if cohort(entry) == "cluster"],
                    ),
                    "controls": probe_gradient(
                        model,
                        [entry for entry in entries if cohort(entry) == "controls"],
                    ),
                }
                model.train(was_training)
                pending["raw"] = raw
            return
        before = pending.pop("before")
        after = tensor_snapshot(model)
        step = pending.pop("step")
        row = {
            "optimizer_step": step,
            "epoch": context["epoch"],
            "batch_indexes": [int(value) for value in context["batch_indexes"]],
            "batch_hash": batch_hash(context["batch_indexes"]),
            "batch_size": len(context["batch_indexes"]),
        }
        if instrumented:
            before_model = model_from_snapshot(before, x.shape[1])
            was_training = model.training
            model.eval()
            state_rows, effects = evaluate_step(before_model, model, entries)
            model.train(was_training)
            raw = pending.pop("raw")
            update = input_vector(after) - input_vector(before)
            probe = pending.pop("probe")
            row |= effects | {
                "states": state_rows,
                "input_gradient": {
                    "preclip_norm": float(torch.linalg.vector_norm(raw)),
                    "postclip_norm": float(torch.linalg.vector_norm(raw))
                    * float(context["clip_scale"]),
                    "update_norm": float(torch.linalg.vector_norm(update)),
                    "cosine_update_to_negative_raw_gradient": cosine(update, -raw),
                    "cluster_probe_alignment": cosine(raw, probe["cluster"]),
                    "control_probe_alignment": cosine(raw, probe["controls"]),
                },
            }
            snapshots[step] = {
                "before": before,
                "after": after,
                "optimizer": pending.pop("optimizer"),
            }
        else:
            pending.pop("optimizer")
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
        step_callback=callback if instrumented else None,
        step_callback_needs_raw_gradients=instrumented,
    )
    checkpoint_path = (
        workdir / ("instrumented" if instrumented else "reference") / f"{seed_name}.npz"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint_path, **checkpoint_from_model(model))
    return {
        "model": model,
        "sha": sha256_file(checkpoint_path),
        "traces": traces,
        "snapshots": snapshots,
        "x": x,
        "p": p,
        "v": v,
        "metadata": replay_metadata(inputs, p, v),
    }


def counterfactuals(
    lanes: dict[str, Any], manifest: dict[str, Any], selected: list[int]
) -> tuple[list[dict[str, Any]], bool]:
    result, valid = [], True
    for step in selected:
        matched = min(lanes["T63"]["snapshots"], key=lambda value: abs(value - step))
        pairs = {
            "T61_historical": ("T61", step, "T61", step),
            "T61_matched_T63_batch": ("T61", step, "T63", matched),
            "T63_T61_batch": ("T63", matched, "T61", step),
            "T63_historical": ("T63", matched, "T63", matched),
        }
        cells = {}
        for name, (recipient, recipient_step, donor, donor_step) in pairs.items():
            recipient_data, donor_data = lanes[recipient], lanes[donor]
            indexes = donor_data["traces"][donor_step - 1]["batch_indexes"]
            clone = clone_adam_step(
                recipient_data["snapshots"][recipient_step]["before"],
                recipient_data["snapshots"][recipient_step]["optimizer"],
                donor_data["x"][indexes],
                donor_data["p"][indexes],
                donor_data["v"][indexes],
            )
            before_model = model_from_snapshot(
                recipient_data["snapshots"][recipient_step]["before"],
                recipient_data["x"].shape[1],
            )
            rows, effects = evaluate_step(before_model, clone, manifest["entries"])
            cells[name] = effects | {
                "input_update_norm": float(
                    torch.linalg.vector_norm(
                        input_vector(tensor_snapshot(clone))
                        - input_vector(
                            recipient_data["snapshots"][recipient_step]["before"]
                        )
                    )
                ),
                "full_policy_margin_change": statistics.fmean(
                    [
                        margin_from_a0(
                            clone,
                            residual_v3_activations(
                                clone,
                                torch.tensor(
                                    [
                                        encode_state(
                                            entry["state"], input_encoding="kalah_v3"
                                        )
                                    ],
                                    dtype=torch.float32,
                                ),
                            )["A0"],
                            entry,
                        )
                        - margin_from_a0(
                            before_model,
                            residual_v3_activations(
                                before_model,
                                torch.tensor(
                                    [
                                        encode_state(
                                            entry["state"], input_encoding="kalah_v3"
                                        )
                                    ],
                                    dtype=torch.float32,
                                ),
                            )["A0"],
                            entry,
                        )
                        for entry in manifest["entries"]
                        if cohort(entry) == "cluster"
                    ]
                ),
                "state_rows": rows,
            }
            if name in {"T61_historical", "T63_historical"}:
                actual = recipient_data["snapshots"][recipient_step]["after"]
                error = max(
                    float((parameter.detach().cpu() - actual[key]).abs().max())
                    for key, parameter in clone.named_parameters()
                )
                cells[name]["historical_max_abs_tensor_error"] = error
                valid &= error <= CLONE_TOLERANCE
        result.append({"t61_step": step, "matched_t63_step": matched, "cells": cells})
    return result, valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-result", type=Path, required=True)
    parser.add_argument("--out-top-batches", type=Path, required=True)
    parser.add_argument("--out-counterfactuals", type=Path, required=True)
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
        "formation_window": [1, FORMATION_END],
        "guardrails": {
            "self_play": False,
            "replay_mutation": False,
            "training_intervention": False,
            "promotion": False,
            "frozen_cluster_training_ineligible": True,
        },
    }
    if not args.execute:
        result |= {"classification": "planned", "next_experiment": "not run"}
        write_json(args.out_result, result)
        return 0
    lanes, baseline = {}, {}
    for seed in TRAINING_SEEDS:
        reference, live = (
            run_lane(paths, manifest, seed, args.workdir, False),
            run_lane(paths, manifest, seed, args.workdir, True),
        )
        parity = reference["sha"] == live["sha"] == EXPECTED_SHA[seed]
        baseline[seed] = {
            "expected_sha256": EXPECTED_SHA[seed],
            "reference_sha256": reference["sha"],
            "instrumented_sha256": live["sha"],
            "reproduced": parity,
        }
        lanes[seed] = live
    if not all(item["reproduced"] for item in baseline.values()):
        result |= {
            "baseline_reproduction": baseline,
            "classification": "early_trunk_provenance_baseline_not_reproduced",
            "next_experiment": "none",
        }
        write_json(args.out_result, result)
        raise RuntimeError("early_trunk_provenance_baseline_not_reproduced")
    formation = {
        seed: [row for row in lane["traces"] if row["optimizer_step"] <= FORMATION_END]
        for seed, lane in lanes.items()
    }
    rankings = {seed: ranked_steps(rows) for seed, rows in formation.items()}
    top = {
        "T61": {
            key: [
                next(row for row in formation["T61"] if row["optimizer_step"] == step)
                for step in steps
            ]
            for key, steps in rankings["T61"].items()
            if isinstance(steps, list)
        }
    }
    cells, valid = counterfactuals(
        lanes,
        manifest,
        [row["optimizer_step"] for row in top["T61"]["worst_10"]],
    )
    classification = (
        "early_trunk_batch_counterfactual_invalid"
        if not valid
        else "early_trunk_provenance_heterogeneous"
    )
    next_experiment = (
        "none"
        if not valid
        else "restrict analysis to the earliest pre-step-108 window in which cumulative T61-vs-T63 A0 damage first becomes material, and repeat the 2x2 batch-state counterfactual there only."
    )
    result |= {
        "baseline_reproduction": baseline,
        "replay_rows": lanes["T61"]["metadata"],
        "unique_compact_rows": len(lanes["T61"]["metadata"]),
        "effective_row_exposures": len(
            load_jsonl_replay(
                [paths["replays"][REPLAY], *paths["fixed"]],
                list(REPLAY_WEIGHTS),
                policy_target_mode="sharpened",
                value_target_mode="sharpened",
            )[3]
        ),
        "per_step": {seed: lane["traces"] for seed, lane in lanes.items()},
        "formation_rankings": rankings,
        "counterfactual_valid": valid,
        "classification": classification,
        "next_experiment": next_experiment,
    }
    write_json(args.out_result, result)
    write_json(args.out_top_batches, top)
    write_json(
        args.out_counterfactuals,
        {"cells": cells, "historical_reproduction_valid": valid},
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        "\n".join(
            [
                "# R61 Early-Trunk Replay Provenance Audit",
                "",
                "Inherited #319 classification: `cluster_representation_drift_early_trunk_primary` at A0.",
                "",
                "## Baseline SHA Parity",
                "",
                "```json",
                json.dumps(baseline, indent=2),
                "```",
                "",
                "## Formation Window",
                "",
                "Steps 1-108 are pre-registered as formation. Per-step ordered batch provenance, compact-row source mapping, A0 pre/post vectors and movement, fixed-downstream margin effects, input gradients, and probe alignments are in the JSON artifact.",
                "",
                "## Ranking",
                "",
                "```json",
                json.dumps(rankings, indent=2),
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
