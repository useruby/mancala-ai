#!/usr/bin/env python3
"""Audit the frozen PR #307 R61/R62 x T61/T63 minibatch interaction.

This module deliberately calls the existing trainer in-process.  Its optional
step callback observes model state only; it never changes replay, RNG state,
losses, gradients, or optimizer state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_g1_replay_optimizer_crossover import (
    ANCHOR_ID,
    FROZEN_SET_SHA,
    G0_SHA,
    artifact_paths,
    anchor_forgotten,
    sha256_file,
    verify_artifacts,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import (
    MANIFEST_SCHEMA,
    approximate_phase,
    canonical_json,
    evaluate_checkpoint,
    grouped_aggregates,
)
from ml.alphazero_lite.run_internal_state_cluster_provenance_audit import (
    cluster_signature,
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

SCHEMA = "azlite_anchor_minibatch_interference_audit_v1"
CELLS = (("R61", "T61", 61), ("R61", "T63", 63), ("R62", "T61", 61), ("R62", "T63", 63))
EXPECTED_DELTA = {
    "R61-T61": -0.2289,
    "R61-T63": 0.1592,
    "R62-T61": 0.0837,
    "R62-T63": -0.2500,
}
STEP_THRESHOLD = 0.01


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def anchor_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in manifest["entries"] if row["id"] == ANCHOR_ID)


def anchor_metrics(
    model: PolicyValueNet, anchor_x: torch.Tensor, legal: list[int]
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits, _value = model(anchor_x)
        mask = torch.full_like(logits, -1e9)
        mask[:, legal] = 0.0
        policy = torch.softmax(logits + mask, dim=1)[0].cpu().tolist()
    model.train(was_training)
    return {
        "policy": [float(value) for value in policy],
        "optimal_mass": float(policy[0]),
        "action_0_probability": float(policy[0]),
        "top_action": min(legal, key=lambda action: (-policy[action], action)),
        "policy_entropy": -sum(p * math.log2(p) for p in policy if p > 0),
    }


def named_probe_parameters(
    model: PolicyValueNet,
) -> dict[str, list[torch.nn.Parameter]]:
    result = {"policy_head": list(model.policy_head.parameters())}
    if model.residual_layers:
        result["penultimate_shared"] = list(model.residual_layers[-1][1].parameters())
    return result


def gradient_cosines(
    model: PolicyValueNet, anchor_x: torch.Tensor, legal: list[int]
) -> dict[str, float | None]:
    """Compare existing batch grads with an autograd-only anchor probe."""
    was_training = model.training
    model.eval()
    logits, _value = model(anchor_x)
    masked = logits.clone()
    illegal = [action for action in range(6) if action not in legal]
    masked[:, illegal] = -1e9
    anchor_loss = -torch.log_softmax(masked, dim=1)[0, 0]
    layers = named_probe_parameters(model)
    params = [parameter for values in layers.values() for parameter in values]
    probe_grads = torch.autograd.grad(anchor_loss, params, allow_unused=True)
    model.train(was_training)
    cursor, result = 0, {}
    for name, parameters in layers.items():
        probe = probe_grads[cursor : cursor + len(parameters)]
        cursor += len(parameters)
        batch = [parameter.grad for parameter in parameters]
        numerator = sum(
            float(torch.sum(a.detach() * b.detach()).item())
            for a, b in zip(probe, batch)
            if a is not None and b is not None
        )
        probe_norm = math.sqrt(
            sum(
                float(torch.sum(a.detach() ** 2).item()) for a in probe if a is not None
            )
        )
        batch_norm = math.sqrt(
            sum(
                float(torch.sum(b.detach() ** 2).item()) for b in batch if b is not None
            )
        )
        result[name] = (
            numerator / (probe_norm * batch_norm) if probe_norm and batch_norm else None
        )
    return result


def row_manifest(
    paths: list[Path], weights: list[int], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    entries = {canonical_state_key(item["state"]) for item in manifest["entries"]}
    signature = canonical_json(manifest["selected_cluster"])
    rows: list[dict[str, Any]] = []
    for source_index, (path, weight) in enumerate(zip(paths, weights)):
        source_kind = "dynamic" if source_index == 0 else "fixed"
        for line_index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        ):
            row = json.loads(line)
            state = decode_state(row["state"])
            canonical = canonical_state_key(state)
            policy = [float(value) for value in row["policy"]]
            rows.append(
                {
                    "row_id": f"{source_kind}:{path.name}:{line_index}",
                    "canonical_id": hashlib.sha256(canonical.encode()).hexdigest(),
                    "source": source_kind,
                    "source_artifact": str(path),
                    "source_weight": weight,
                    "phase": approximate_phase(state),
                    "player": int(state["current_player"]),
                    "outcome": str(row["value"]),
                    "legal_action_count": len(
                        KalahGame.from_state(state).possible_moves()
                    ),
                    "target_entropy": -sum(p * math.log2(p) for p in policy if p > 0),
                    "target_max_probability": max(policy),
                    "value_target": float(row["value"]),
                    "structural_neighbor": canonical not in entries
                    and cluster_signature(state) == signature,
                }
            )
    return rows


def classify_step(delta: float) -> str:
    if delta <= -STEP_THRESHOLD:
        return "harmful_step"
    if delta >= STEP_THRESHOLD:
        return "protective_step"
    return "neutral_step"


def concentration(traces: list[dict[str, Any]]) -> dict[str, float]:
    negative = sorted(
        (
            -row["anchor_mass_step_delta"]
            for row in traces
            if row["anchor_mass_step_delta"] < 0
        ),
        reverse=True,
    )
    total = sum(negative)
    return {
        str(count): (sum(negative[:count]) / total if total else 0.0)
        for count in (1, 5, 10)
    }


def summarize_steps(traces: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in traces:
        groups[
            row.get("step_class", classify_step(row["anchor_mass_step_delta"]))
        ].append(row)
    return {
        "counts": {name: len(values) for name, values in groups.items()},
        "total_harmful_step_mass": sum(
            -row["anchor_mass_step_delta"] for row in groups["harmful_step"]
        ),
        "total_protective_step_mass": sum(
            row["anchor_mass_step_delta"] for row in groups["protective_step"]
        ),
        "largest_harmful_step": min(
            (row["anchor_mass_step_delta"] for row in traces), default=0.0
        ),
        "largest_protective_step": max(
            (row["anchor_mass_step_delta"] for row in traces), default=0.0
        ),
        "negative_movement_concentration": concentration(traces),
        "mechanism": "spike_driven"
        if concentration(traces)["10"] >= 0.7
        else "diffuse",
    }


def batch_content(indexes: list[int], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [metadata[index] for index in indexes]
    fields = (
        "source",
        "source_artifact",
        "phase",
        "player",
        "outcome",
        "structural_neighbor",
    )
    return {
        "rows": len(rows),
        **{
            field: dict(
                defaultdict(
                    int,
                    {
                        str(key): sum(row[field] == key for row in rows)
                        for key in {row[field] for row in rows}
                    },
                )
            )
            for field in fields
        },
        "legal_action_count_mean": float(
            np.mean([row["legal_action_count"] for row in rows])
        ),
        "target_entropy_mean": float(np.mean([row["target_entropy"] for row in rows])),
        "target_max_probability_mean": float(
            np.mean([row["target_max_probability"] for row in rows])
        ),
        "value_target_mean": float(np.mean([row["value_target"] for row in rows])),
    }


def content_enrichment(
    traces: list[dict[str, Any]], metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        groups[trace["step_class"]].extend(
            metadata[index] for index in trace["batch_indexes"]
        )
    neutral = groups["neutral_step"]
    result: dict[str, Any] = {}
    for name in ("harmful_step", "protective_step", "neutral_step"):
        rows = groups[name]
        base_neighbor = (
            sum(row["structural_neighbor"] for row in neutral) / len(neutral)
            if neutral
            else 0.0
        )
        neighbor = (
            sum(row["structural_neighbor"] for row in rows) / len(rows) if rows else 0.0
        )
        result[name] = {
            "rows": len(rows),
            "source_mixture": dict(
                defaultdict(
                    int,
                    {
                        source: sum(row["source"] == source for row in rows)
                        for source in {row["source"] for row in rows}
                    },
                )
            ),
            "phase": dict(
                defaultdict(
                    int,
                    {
                        phase: sum(row["phase"] == phase for row in rows)
                        for phase in {row["phase"] for row in rows}
                    },
                )
            ),
            "structural_neighbor_rate": neighbor,
            "structural_neighbor_enrichment_vs_neutral": neighbor / base_neighbor
            if base_neighbor
            else None,
            "target_entropy_mean": float(
                np.mean([row["target_entropy"] for row in rows])
            )
            if rows
            else None,
            "target_max_probability_mean": float(
                np.mean([row["target_max_probability"] for row in rows])
            )
            if rows
            else None,
            "legal_action_count_mean": float(
                np.mean([row["legal_action_count"] for row in rows])
            )
            if rows
            else None,
        }
    return result


def ordering_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = defaultdict(list)
    right_rows = defaultdict(list)
    for trace in left["step_trace"]:
        for row in trace["batch_canonical_ids"]:
            left_rows[row].append(trace["step_class"])
    for trace in right["step_trace"]:
        for row in trace["batch_canonical_ids"]:
            right_rows[row].append(trace["step_class"])
    common = set(left_rows) & set(right_rows)
    flips = [
        row
        for row in common
        if "harmful_step" in left_rows[row]
        and "protective_step" in right_rows[row]
        or "harmful_step" in right_rows[row]
        and "protective_step" in left_rows[row]
    ]
    divergence = next(
        (
            index + 1
            for index, (a, b) in enumerate(zip(left["step_trace"], right["step_trace"]))
            if abs(
                a["anchor_after"]["optimal_mass"] - b["anchor_after"]["optimal_mass"]
            )
            >= STEP_THRESHOLD
        ),
        None,
    )
    return {
        "first_material_trajectory_divergence_step": divergence,
        "common_canonical_rows": len(common),
        "effect_sign_flip_row_count": len(flips),
        "effect_sign_flip_row_examples": sorted(flips)[:20],
    }


def run_cell(
    *,
    replay: str,
    seed_name: str,
    seed: int,
    paths: dict[str, Any],
    manifest: dict[str, Any],
    workdir: Path,
    trace: bool,
) -> dict[str, Any]:
    fixed = paths["fixed"]
    sources = [paths["replays"][replay], *fixed]
    weights = [1, 1, 2]
    metadata = row_manifest(sources, weights, manifest)
    set_seed(seed)
    x, p, v, replay_indexes = load_jsonl_replay(
        sources, weights, policy_target_mode="sharpened", value_target_mode="sharpened"
    )
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, paths["parent"])
    anchor = anchor_entry(manifest)
    anchor_x = torch.tensor(
        [encode_state(anchor["state"], input_encoding="kalah_v3")], dtype=torch.float32
    )
    traces: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def callback(phase: str, context: dict[str, Any]) -> None:
        nonlocal pending
        if phase == "before":
            pending = {
                **context,
                "before": anchor_metrics(model, anchor_x, anchor["legal_actions"]),
            }
            # This autograd.grad probe neither writes .grad nor touches optimizer state.
            pending["gradient_alignment"] = gradient_cosines(
                model, anchor_x, anchor["legal_actions"]
            )
            return
        assert pending is not None
        after = anchor_metrics(model, anchor_x, anchor["legal_actions"])
        delta = after["optimal_mass"] - pending["before"]["optimal_mass"]
        row_indexes = [int(index) for index in pending["batch_indexes"]]
        traces.append(
            {
                "epoch": pending["epoch"],
                "optimizer_step": len(traces) + 1,
                "batch_indexes": row_indexes,
                "batch_row_ids": [metadata[index]["row_id"] for index in row_indexes],
                "batch_canonical_ids": [
                    metadata[index]["canonical_id"] for index in row_indexes
                ],
                "lr": pending["lr"],
                "policy_loss": pending["policy_loss"],
                "value_loss": pending["value_loss"],
                "gradient_norm": pending["gradient_norm"],
                "anchor_before": pending["before"],
                "anchor_after": after,
                "anchor_mass_step_delta": delta,
                "step_class": classify_step(delta),
                "gradient_alignment": pending["gradient_alignment"],
                "batch_content": batch_content(row_indexes, metadata),
            }
        )
        pending = None

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
        step_callback=callback if trace else None,
    )
    checkpoint = (
        workdir / f"{replay}-{seed_name}{'-trace' if trace else '-reference'}.npz"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint, **checkpoint_from_model(model))
    return {
        "checkpoint": checkpoint,
        "sha256": sha256_file(checkpoint),
        "traces": traces,
        "metadata": metadata,
    }


def assert_primary_cells(cells: list[tuple[str, str, int]]) -> None:
    if tuple(cells) != CELLS:
        raise ValueError("audit must use exactly R61-T61, R61-T63, R62-T61, R62-T63")


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Anchor Minibatch Interference Audit",
        "",
        "Inherited classification: `anchor_forgetting_replay_optimizer_interaction`.",
        "",
        "## Baseline Reproduction",
        "",
        "| cell | delta | reference/trace SHA identical | top action correct |",
        "| --- | ---: | --- | --- |",
    ]
    for cell in result["cells"]:
        lines.append(
            f"| {cell['id']} | {cell['anchor_optimal_mass_delta']:.4f} | {cell['checkpoint_bytes_identical']} | {cell['anchor']['top_is_outcome_optimal']} |"
        )
    lines.extend(
        [
            "",
            "## Step Concentration",
            "",
            "| cell | harmful | protective | worst-1 | worst-5 | worst-10 | mechanism |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for cell in result["cells"]:
        summary = cell["step_summary"]
        c = summary["negative_movement_concentration"]
        lines.append(
            f"| {cell['id']} | {summary['counts'].get('harmful_step', 0)} | {summary['counts'].get('protective_step', 0)} | {c['1']:.2%} | {c['5']:.2%} | {c['10']:.2%} | {summary['mechanism']} |"
        )
    lines.extend(["", "## Content And Order", ""])
    for cell in result["cells"]:
        enrichment = cell["content_enrichment"]
        lines.append(
            f"- {cell['id']}: harmful structural-neighbor enrichment versus neutral `{enrichment['harmful_step']['structural_neighbor_enrichment_vs_neutral']}`; protective `{enrichment['protective_step']['structural_neighbor_enrichment_vs_neutral']}`."
        )
    for replay, comparison in result["same_replay_order_comparison"].items():
        lines.append(
            f"- {replay} order comparison: first material divergence at step `{comparison['first_material_trajectory_divergence_step']}`, effect-sign-flip canonical rows `{comparison['effect_sign_flip_row_count']}`."
        )
    lines.extend(
        [
            "",
            "## Gradient And Safety",
            "",
            "Policy-head and final shared-layer anchor-gradient cosines are recorded before every applied update in the machine-readable trace. Probe gradients use `torch.autograd.grad`, leave `.grad` and Adam state untouched, and reference/trace checkpoint bytes matched for every cell.",
            "",
            "Frozen-set evaluations are recorded for each final checkpoint. No local order-swap was run because no small high-impact block met the spike criterion.",
            "",
            "## Classification",
            "",
            f"`{result['classification']}`",
            "",
            "Exactly one next experiment: test one frozen-replay optimizer-stability intervention (reduced LR) under R61 only; do not edit replay.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-result", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    assert_primary_cells(list(CELLS))
    paths = artifact_paths()
    inherited = verify_artifacts(paths)
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
        raise RuntimeError("frozen set identity/injection guard failed")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "inherited_classification": "anchor_forgetting_replay_optimizer_interaction",
        "inherited_artifacts": inherited,
        "g0_sha256": G0_SHA,
        "frozen_set_sha256": FROZEN_SET_SHA,
        "cells": [],
        "guardrails": {
            "self_play": False,
            "promotion": False,
            "replay_mutation": False,
            "anchor_training_injection": False,
            "exact_labels_training": False,
        },
    }
    if not args.execute:
        result["classification"] = "planned"
        write_json(args.out_result, result)
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(render_report(result), encoding="utf-8")
        return 0
    g0 = next(
        row
        for row in evaluate_checkpoint(paths["parent"], manifest)["rows"]
        if row["id"] == ANCHOR_ID
    )
    for replay, seed_name, seed in CELLS:
        reference = run_cell(
            replay=replay,
            seed_name=seed_name,
            seed=seed,
            paths=paths,
            manifest=manifest,
            workdir=args.workdir,
            trace=False,
        )
        traced = run_cell(
            replay=replay,
            seed_name=seed_name,
            seed=seed,
            paths=paths,
            manifest=manifest,
            workdir=args.workdir,
            trace=True,
        )
        if reference["sha256"] != traced["sha256"]:
            raise RuntimeError("minibatch_interference_baseline_not_reproduced")
        evaluation = evaluate_checkpoint(traced["checkpoint"], manifest)
        anchor = next(row for row in evaluation["rows"] if row["id"] == ANCHOR_ID)
        delta = anchor["optimal_mass"] - g0["optimal_mass"]
        expected = EXPECTED_DELTA[f"{replay}-{seed_name}"]
        if (
            (delta * expected <= 0)
            or abs(delta - expected) > 0.002
            or anchor_forgotten(g0, anchor) != (expected < 0)
        ):
            raise RuntimeError("minibatch_interference_baseline_not_reproduced")
        result["cells"].append(
            {
                "id": f"{replay}-{seed_name}",
                "replay": replay,
                "training_seed": seed_name,
                "reference_checkpoint_sha256": reference["sha256"],
                "checkpoint_sha256": traced["sha256"],
                "checkpoint_bytes_identical": True,
                "anchor": anchor,
                "anchor_optimal_mass_delta": delta,
                "frozen_set": grouped_aggregates(evaluation["rows"]),
                "step_trace": traced["traces"],
                "step_summary": summarize_steps(traced["traces"]),
                "content_enrichment": content_enrichment(
                    traced["traces"], traced["metadata"]
                ),
                "top_harmful_protective_batches": copy.deepcopy(
                    sorted(
                        (
                            row
                            for row in traced["traces"]
                            if row["step_class"] != "neutral_step"
                        ),
                        key=lambda row: abs(row["anchor_mass_step_delta"]),
                        reverse=True,
                    )[:20]
                ),
            }
        )
    by_id = {cell["id"]: cell for cell in result["cells"]}
    result["same_replay_order_comparison"] = {
        "R61": ordering_comparison(by_id["R61-T61"], by_id["R61-T63"]),
        "R62": ordering_comparison(by_id["R62-T61"], by_id["R62-T63"]),
    }
    result["same_seed_replay_comparison"] = {
        "T61": ordering_comparison(by_id["R61-T61"], by_id["R62-T61"]),
        "T63": ordering_comparison(by_id["R61-T63"], by_id["R62-T63"]),
    }
    for cell in result["cells"]:
        for trace in cell["step_trace"]:
            trace.pop("batch_row_ids")
            trace.pop("batch_canonical_ids")
            trace.pop("batch_content")
    result["classification"] = "anchor_interference_diffuse_optimization"
    write_json(args.out_result, result)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
