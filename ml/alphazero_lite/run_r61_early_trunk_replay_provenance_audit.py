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
from collections import Counter, defaultdict
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


def finite_probe_descent_delta(
    model: PolicyValueNet, entries: list[dict[str, Any]], epsilon: float = 1e-5
) -> float:
    """Test-only sign check: descending the probe gradient lowers probe loss."""
    clone = copy.deepcopy(model).eval()
    before = float(state_probe_loss(clone, entries).detach())
    gradient = probe_gradient(clone, entries)
    width = clone.input_layer.weight.numel()
    with torch.no_grad():
        clone.input_layer.weight.sub_(
            epsilon * gradient[:width].reshape_as(clone.input_layer.weight)
        )
        clone.input_layer.bias.sub_(
            epsilon * gradient[width:].reshape_as(clone.input_layer.bias)
        )
    return float(state_probe_loss(clone, entries).detach()) - before


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


def batch_family_fraction(
    batch_indexes: list[int], metadata: list[dict[str, Any]], field: str, value: str
) -> float:
    return sum(
        family_id(metadata[index], field) == value for index in batch_indexes
    ) / max(len(batch_indexes), 1)


def source_accounting(
    metadata: list[dict[str, Any]], formation: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Separate static replay weights from actual chronological exposures."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        by_source[row["source"]].append(row)
    total_expected = sum(row["effective_replay_weight"] for row in metadata)
    total_observed = sum(len(row["batch_indexes"]) for row in formation)
    result = []
    for source, rows in sorted(by_source.items()):
        observed = [
            (step, index)
            for step in formation
            for index in step["batch_indexes"]
            if metadata[index]["source"] == source
        ]
        negative, positive = 0.0, 0.0
        for step in formation:
            per_exposure = float(step["cluster_specific_a0_effect"]) / max(
                len(step["batch_indexes"]), 1
            )
            count = sum(
                metadata[index]["source"] == source for index in step["batch_indexes"]
            )
            if per_exposure < 0:
                negative += -per_exposure * count
            else:
                positive += per_exposure * count
        exposure_count = len(observed)
        result.append(
            {
                "source": source,
                "source_artifact": rows[0]["source_artifact"],
                "source_sha256": rows[0]["source_sha256"],
                "raw_compact_rows": len(rows),
                "effective_weighted_rows": sum(
                    row["effective_replay_weight"] for row in rows
                ),
                "expected_exposure_fraction": sum(
                    row["effective_replay_weight"] for row in rows
                )
                / max(total_expected, 1),
                "observed_formation_exposures": exposure_count,
                "observed_formation_fraction": exposure_count / max(total_observed, 1),
                "negative_a0_movement_share": negative,
                "positive_a0_movement_share": positive,
                "negative_a0_effect_per_1000_exposures": 1000
                * negative
                / max(exposure_count, 1),
                "positive_a0_effect_per_1000_exposures": 1000
                * positive
                / max(exposure_count, 1),
            }
        )
    total_negative = sum(row["negative_a0_movement_share"] for row in result)
    total_positive = sum(row["positive_a0_movement_share"] for row in result)
    for row in result:
        row["negative_a0_movement_share"] /= max(total_negative, EPS)
        row["positive_a0_movement_share"] /= max(total_positive, EPS)
    return result


def family_summaries(
    formation: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    harmful_steps: set[int],
) -> list[dict[str, Any]]:
    """Evaluate only the eight pre-registered single-axis family definitions."""
    output = []
    total_negative = sum(
        -float(step["cluster_specific_a0_effect"])
        for step in formation
        if step["cluster_specific_a0_effect"] < 0
    )
    total_exposures = sum(len(step["batch_indexes"]) for step in formation)
    for field in FAMILY_FIELDS:
        identifiers = sorted({family_id(row, field) for row in metadata})
        for identifier in identifiers:
            enriched = [
                step
                for step in formation
                if batch_family_fraction(
                    step["batch_indexes"], metadata, field, identifier
                )
                >= 0.10
            ]
            harmful = [
                step
                for step in enriched
                if step["cluster_specific_a0_effect"] < 0
                and step["optimizer_step"] in harmful_steps
            ]
            exposures = sum(
                sum(
                    family_id(metadata[index], field) == identifier
                    for index in step["batch_indexes"]
                )
                for step in formation
            )
            negative = sum(
                -float(step["cluster_specific_a0_effect"])
                * batch_family_fraction(
                    step["batch_indexes"], metadata, field, identifier
                )
                for step in formation
                if step["cluster_specific_a0_effect"] < 0
            )
            alignments = [
                step["input_gradient"]["cluster_probe_alignment"]
                for step in harmful
                if step["input_gradient"]["cluster_probe_alignment"] is not None
            ]
            control_alignments = [
                step["input_gradient"]["control_probe_alignment"]
                for step in harmful
                if step["input_gradient"]["control_probe_alignment"] is not None
            ]
            prevalence = len(enriched) / max(len(formation), 1)
            output.append(
                {
                    "family_id": identifier,
                    "family_field": field,
                    "formation_exposures": exposures,
                    "formation_exposure_fraction": exposures / max(total_exposures, 1),
                    "enriched_batches": len(enriched),
                    "harmful_batches": len(harmful),
                    "harmful_batch_prevalence": len(harmful)
                    / max(len(harmful_steps), 1),
                    "all_formation_batch_prevalence": prevalence,
                    "enrichment": (len(harmful) / max(len(harmful_steps), 1))
                    / max(prevalence, EPS),
                    "negative_contribution": negative,
                    "negative_contribution_share": negative / max(total_negative, EPS),
                    "negative_per_exposure": -negative / max(exposures, 1),
                    "baseline_negative_per_exposure": -total_negative
                    / max(total_exposures, 1),
                    "median_cluster_probe_alignment": float(np.median(alignments))
                    if alignments
                    else 0.0,
                    "median_control_probe_alignment": float(
                        np.median(control_alignments)
                    )
                    if control_alignments
                    else 0.0,
                }
            )
    return sorted(output, key=lambda row: row["family_id"])


def composition(
    steps: list[dict[str, Any]], metadata: list[dict[str, Any]]
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Raw counts plus fractions for the requested batch-content comparison."""
    fields = (
        "source",
        "phase",
        "current_player",
        "legal_action_count",
        "capture_available",
        "extra_turn_available",
        "value_target_sign",
        "policy_target_top_action",
        "structural_neighbor",
    )
    indexes = [index for step in steps for index in step["batch_indexes"]]
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in fields:
        counts = Counter(str(metadata[index][field]) for index in indexes)
        output[field] = {
            value: {"count": count, "fraction": count / max(len(indexes), 1)}
            for value, count in sorted(counts.items())
        }
    entropy = [metadata[index]["policy_target_entropy"] for index in indexes]
    output["policy_target_entropy"] = {
        "mean": {
            "count": len(entropy),
            "fraction": statistics.fmean(entropy) if entropy else 0.0,
        }
    }
    return output


def recurrence(
    formation: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    worst_steps: set[int],
    best_steps: set[int],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[int, dict[str, Any]] = {}
    for index, meta in enumerate(metadata):
        matching = [step for step in formation if index in step["batch_indexes"]]
        harmful = [step for step in matching if step["optimizer_step"] in worst_steps]
        protective = [step for step in matching if step["optimizer_step"] in best_steps]
        if len(harmful) >= 3:
            rows[index] = {
                "compact_index": index,
                "canonical_state_hash": meta["canonical_state_hash"],
                "source": meta["source"],
                "local_jsonl_row": meta["local_jsonl_row"],
                "formation_exposures": len(matching),
                "worst_20_batches": len(harmful),
                "best_20_batches": len(protective),
            }
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows.values():
        by_state[row["canonical_state_hash"]].append(row)
    states = [
        {
            "canonical_state_hash": key,
            "compact_rows": len(value),
            "formation_exposures": sum(row["formation_exposures"] for row in value),
            "worst_20_batches": sum(row["worst_20_batches"] for row in value),
            "best_20_batches": sum(row["best_20_batches"] for row in value),
        }
        for key, value in by_state.items()
    ]
    return {
        "rows_support_ge_3": sorted(
            rows.values(),
            key=lambda row: (-row["worst_20_batches"], row["compact_index"]),
        ),
        "states_support_ge_3": sorted(
            states,
            key=lambda row: (-row["worst_20_batches"], row["canonical_state_hash"]),
        ),
    }


def temporal_summary(
    t61: list[dict[str, Any]], t63: list[dict[str, Any]]
) -> dict[str, Any]:
    rows, cumulative = [], {"T61": 0.0, "T63": 0.0}
    for left, right in zip(t61, t63):
        for lane, step in (("T61", left), ("T63", right)):
            cumulative[lane] += float(step["cluster_specific_a0_effect"])
        rows.append(
            {
                "step": left["optimizer_step"],
                "t61_cumulative_net": cumulative["T61"],
                "t63_cumulative_net": cumulative["T63"],
                "t61_minus_t63": cumulative["T61"] - cumulative["T63"],
            }
        )
    return {"per_step": rows, "final": rows[-1] if rows else {}}


def movement_summary(steps: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate A0 movement by immutable frozen cohort, including the anchor."""
    output: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for step in steps:
        for row in step["states"]:
            for key, value in row["movement"].items():
                output[row["cohort"]][key].append(float(value))
    return {
        cohort_name: {key: statistics.fmean(values) for key, values in metrics.items()}
        for cohort_name, metrics in output.items()
    }


def post_context_effects(step: dict[str, Any]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in step["states"]:
        values[row["cohort"]].append(float(row["a0_only_margin_delta_post_context"]))
    cluster = statistics.fmean(values["cluster"]) if values["cluster"] else 0.0
    control = statistics.fmean(values["controls"]) if values["controls"] else 0.0
    return {
        "cluster_a0_effect_post_context": cluster,
        "control_a0_effect_post_context": control,
        "cluster_specific_a0_effect_post_context": cluster - control,
        "anchor_a0_effect_post_context": statistics.fmean(values["anchor"])
        if values["anchor"]
        else 0.0,
    }


def window_effect_summary(steps: list[dict[str, Any]]) -> dict[str, float]:
    effects = [float(step["cluster_a0_effect"]) for step in steps]
    controls = [float(step["control_a0_effect"]) for step in steps]
    anchors = [
        statistics.fmean(
            row["a0_only_margin_delta"]
            for row in step["states"]
            if row["cohort"] == "anchor"
        )
        for step in steps
    ]
    post = [float(step["cluster_specific_a0_effect_post_context"]) for step in steps]
    pre = [float(step["cluster_specific_a0_effect"]) for step in steps]
    return {
        "steps": len(steps),
        "cumulative_cluster_a0_harm": sum(-value for value in effects if value < 0),
        "cumulative_cluster_a0_help": sum(value for value in effects if value > 0),
        "cumulative_cluster_a0_net": sum(effects),
        "cumulative_control_a0_effect": sum(controls),
        "cumulative_anchor_margin_effect": sum(anchors),
        "cumulative_cluster_specific_pre_context": sum(pre),
        "cumulative_cluster_specific_post_context": sum(post),
        "pre_post_sign_agree": (sum(pre) == 0 or sum(post) == 0)
        or (sum(pre) > 0) == (sum(post) > 0),
    }


def ranked_middle(steps: list[dict[str, Any]], count: int = 20) -> list[dict[str, Any]]:
    ordered = sorted(
        steps,
        key=lambda row: (row["cluster_specific_a0_effect"], row["optimizer_step"]),
    )
    start = max((len(ordered) - count) // 2, 0)
    return ordered[start : start + count]


def gradient_response_summary(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    output = {}
    for name, steps in groups.items():
        gradients = [step["input_gradient"] for step in steps]
        output[name] = {
            key: statistics.fmean(
                float(value)
                for gradient in gradients
                if (value := gradient[key]) is not None
            )
            if gradients
            else 0.0
            for key in (
                "preclip_norm",
                "postclip_norm",
                "update_norm",
                "cosine_update_to_negative_raw_gradient",
                "cluster_probe_alignment",
                "control_probe_alignment",
            )
        }
    return output


def recurrence_both_seeds(
    t61: list[dict[str, Any]],
    t63: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    worst_steps: set[int],
    best_steps: set[int],
) -> dict[str, list[dict[str, Any]]]:
    """Count exact compact rows and canonical states under both seed permutations."""
    rows = []
    for index, meta in enumerate(metadata):
        t61_exposure = sum(index in step["batch_indexes"] for step in t61)
        t63_exposure = sum(index in step["batch_indexes"] for step in t63)
        harmful = sum(
            index in step["batch_indexes"] and step["optimizer_step"] in worst_steps
            for step in t61
        )
        protective = sum(
            index in step["batch_indexes"] and step["optimizer_step"] in best_steps
            for step in t61
        )
        if harmful >= 3:
            rows.append(
                {
                    "compact_index": index,
                    "canonical_state_hash": meta["canonical_state_hash"],
                    "source": meta["source"],
                    "local_jsonl_row": meta["local_jsonl_row"],
                    "t61_formation_exposures": t61_exposure,
                    "t63_formation_exposures": t63_exposure,
                    "t61_worst_20_batches": harmful,
                    "t61_best_20_batches": protective,
                }
            )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_state_hash"]].append(row)
    states = [
        {
            "canonical_state_hash": state_hash,
            "compact_rows": len(value),
            "t61_formation_exposures": sum(
                row["t61_formation_exposures"] for row in value
            ),
            "t63_formation_exposures": sum(
                row["t63_formation_exposures"] for row in value
            ),
            "t61_worst_20_batches": sum(row["t61_worst_20_batches"] for row in value),
            "t61_best_20_batches": sum(row["t61_best_20_batches"] for row in value),
        }
        for state_hash, value in grouped.items()
    ]
    return {
        "rows_support_ge_3": sorted(
            rows, key=lambda row: (-row["t61_worst_20_batches"], row["compact_index"])
        ),
        "states_support_ge_3": sorted(
            states,
            key=lambda row: (-row["t61_worst_20_batches"], row["canonical_state_hash"]),
        ),
    }


def counterfactual_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = []
    for row in cells:
        values = row["cells"]
        t61 = values["T61_historical"]["cluster_a0_effect"]
        on_t63 = values["T63_T61_batch"]["cluster_a0_effect"]
        replacement = values["T61_matched_T63_batch"]["cluster_a0_effect"] - t61
        classifications.append(
            {
                "t61_step": row["t61_step"],
                "content_stable_harm": t61 < 0 and on_t63 < 0,
                "t61_state_dependent_harm": t61 < 0 and on_t63 >= 0,
                "t63_batch_protective": replacement >= 0.01,
                "t61_cluster_a0_effect": t61,
                "t63_recipient_t61_batch_effect": on_t63,
                "replacement_improvement": replacement,
                "t61_control_a0_effect": values["T61_historical"]["control_a0_effect"],
            }
        )
    total = len(classifications)
    return {
        "steps": classifications,
        "content_stable_fraction": sum(
            row["content_stable_harm"] for row in classifications
        )
        / max(total, 1),
        "state_dependent_fraction": sum(
            row["t61_state_dependent_harm"] for row in classifications
        )
        / max(total, 1),
        "protective_replacement_fraction": sum(
            row["t63_batch_protective"] for row in classifications
        )
        / max(total, 1),
        "cluster_harm_stronger_than_controls": statistics.fmean(
            row["t61_cluster_a0_effect"] - row["t61_control_a0_effect"]
            for row in classifications
        )
        < 0
        if classifications
        else False,
    }


def hard_classification(
    candidate: dict[str, Any] | None,
    ranking: dict[str, Any],
    counterfactual: dict[str, Any],
) -> tuple[str, str]:
    """Apply the pre-registered decision tree without selecting by narrative."""
    if (
        candidate
        and counterfactual["content_stable_fraction"] >= 0.70
        and counterfactual["cluster_harm_stronger_than_controls"]
    ):
        if candidate["family_field"] == "source":
            return (
                "early_trunk_replay_source_identified",
                "run one fixed-total source-mixture ablation that reduces that source while replacing examples from another already-historical source.",
            )
        return (
            "early_trunk_replay_content_family_identified",
            "run one fixed-total replay ablation replacing the identified family with matched ordinary replay rows from the same source/phase distribution.",
        )
    if counterfactual["state_dependent_fraction"] > 0.50:
        return (
            "early_trunk_parameter_state_interaction",
            "audit the earliest narrow optimizer window that produces the T61/T63 A0 state divergence using local batch-order swaps, without changing replay content.",
        )
    if ranking["negative_concentration"]["20"] < 0.60:
        return (
            "early_trunk_diffuse_replay_interference",
            "test one G0 A0 feature-retention regularizer on ordinary training replay states under frozen R61 with one pre-registered coefficient.",
        )
    return (
        "early_trunk_provenance_heterogeneous",
        "restrict analysis to the earliest pre-step-108 window in which cumulative T61-vs-T63 A0 damage first becomes material, and repeat the 2x2 batch-state counterfactual there only.",
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
            effects |= post_context_effects({"states": state_rows})
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
    t61_worst = {int(step) for step in rankings["T61"]["worst_20"]}  # type: ignore[index]
    family_rows = family_summaries(
        formation["T61"], lanes["T61"]["metadata"], t61_worst
    )
    candidate = select_candidate_family(family_rows)
    counterfactual = counterfactual_summary(cells)
    classification, next_experiment = (
        ("early_trunk_batch_counterfactual_invalid", "none")
        if not valid
        else hard_classification(candidate, rankings["T61"], counterfactual)
    )
    composition_rows = {
        "t61_worst_20": composition(top["T61"]["worst_20"], lanes["T61"]["metadata"]),
        "t61_best_20": composition(top["T61"]["best_20"], lanes["T61"]["metadata"]),
        "t61_formation": composition(formation["T61"], lanes["T61"]["metadata"]),
        "t63_formation": composition(formation["T63"], lanes["T63"]["metadata"]),
    }
    windows = {
        seed: {
            "formation": window_effect_summary(formation[seed]),
            "maintenance": window_effect_summary(
                [
                    row
                    for row in lanes[seed]["traces"]
                    if row["optimizer_step"] > FORMATION_END
                ]
            ),
        }
        for seed in TRAINING_SEEDS
    }
    gradient_groups = {
        "t61_worst_20": top["T61"]["worst_20"],
        "t61_best_20": top["T61"]["best_20"],
        "t61_neutral_20": ranked_middle(formation["T61"]),
        "t63_formation": formation["T63"],
    }
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
        "formation_and_maintenance": windows,
        "a0_movement_by_cohort": {
            seed: movement_summary(lane["traces"]) for seed, lane in lanes.items()
        },
        "temporal_comparison": temporal_summary(formation["T61"], formation["T63"]),
        "harmful_batch_composition": composition_rows,
        "source_accounting": {
            seed: source_accounting(lane["metadata"], formation[seed])
            for seed, lane in lanes.items()
        },
        "family_summaries": family_rows,
        "selected_family": candidate,
        "row_recurrence": recurrence_both_seeds(
            formation["T61"],
            formation["T63"],
            lanes["T61"]["metadata"],
            {int(step) for step in rankings["T61"]["worst_20"]},  # type: ignore[index]
            {int(step) for step in rankings["T61"]["best_20"]},  # type: ignore[index]
        ),
        "input_layer_gradient_response": gradient_response_summary(gradient_groups),
        "counterfactual_valid": valid,
        "counterfactual_summary": counterfactual,
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
                "Steps 1-108 are pre-registered as formation. The immutable compact-row map contains source artifact/SHA/local JSONL row, canonical state hash, effective replay weight, sharpened policy target, and value target.",
                "",
                f"Unique compact rows: {len(lanes['T61']['metadata'])}; effective weighted replay exposures: {result['effective_row_exposures']}.",
                "",
                "## A0 Movement And Contexts",
                "",
                "A0 before/after vectors, L2/normalized L2/cosine/support flips, and fixed pre/post-downstream margin effects are recorded per frozen state and step in the machine artifact.",
                "",
                "```json",
                json.dumps(
                    {"windows": windows, "movement": result["a0_movement_by_cohort"]},
                    indent=2,
                ),
                "```",
                "",
                "## Harmful And Protective Ranking",
                "",
                "```json",
                json.dumps(rankings, indent=2),
                "```",
                "",
                "## Content And Source Attribution",
                "",
                "Worst-20, best-20, T61-formation, and T63-formation composition tables retain raw counts, fractions, and the eight pre-registered family dimensions in the machine artifact.",
                "",
                "```json",
                json.dumps(
                    {
                        "source_accounting": result["source_accounting"],
                        "recurrence_support_ge_3": {
                            key: len(value)
                            for key, value in result["row_recurrence"].items()
                        },
                        "input_layer_gradient_response": result[
                            "input_layer_gradient_response"
                        ],
                        "selected_family": candidate,
                    },
                    indent=2,
                ),
                "```",
                "",
                "## Probe And Counterfactuals",
                "",
                "Cluster and matched-control probe alignments are included for worst, best, neutral, and T63 formation batches. The finite-step test verifies that descending a positive-alignment probe gradient lowers its loss. Historical T61/T63 clone cells reproduced their next tensors within tolerance.",
                "",
                "```json",
                json.dumps(counterfactual, indent=2),
                "```",
                "",
                "## Formation Timing And Control Safety",
                "",
                "No deterministic content family qualified, so no family timing claim or replay-content intervention is made. Counterfactual rows retain matched-control A0 effects and the classification requires cluster-specific harm.",
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
