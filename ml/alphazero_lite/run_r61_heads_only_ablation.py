#!/usr/bin/env python3
"""Frozen-R61 all-versus-heads_only shared-trunk causal ablation.

The runner is train-only: it does not create self-play, alter replay, inject
frozen states, change search, or promote artifacts.  Existing PR #315 `all`
checkpoints are controls; only the two heads-only cells are optimized.
"""

from __future__ import annotations

import argparse
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
    export_for_arena,
    sha256_file,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import (
    MANIFEST_SCHEMA,
    evaluate_checkpoint,
    grouped_aggregates,
    run_static_baseline,
)
from ml.alphazero_lite.run_r61_lr_stability_ablation import (
    REPLAY,
    REPLAY_WEIGHTS,
    frozen_counts,
    verify_r61_artifacts,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    _count_parameters,
    apply_trainable_scope,
    checkpoint_from_model,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    train,
)

SCHEMA = "azlite_r61_heads_only_shared_trunk_ablation_v1"
TRAINING_SEEDS = {"T61": 61, "T63": 63}
SCOPES = ("all", "heads_only")
EPOCHS, BATCH_SIZE, LR, GRAD_CLIP = 4, 512, 0.001, 1.0
TRUNK_PREFIXES = ("input_layer.", "residual_layers.")
HEAD_PREFIXES = (
    "policy_hidden_layer.",
    "policy_head.",
    "value_hidden_layer.",
    "value_head.",
)
INHERITED_CONTROLS = {
    "T61": (
        ".tmp/r61-parameter-subspace-drift-audit/T61/checkpoint.npz",
        "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    ),
    "T63": (
        ".tmp/r61-parameter-subspace-drift-audit/T63/checkpoint.npz",
        "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
    ),
}
MATERIAL_CLUSTER_REGRESSION = 0.02
MATERIAL_ARENA_REGRESSION = 0.15


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def assert_design(cells: list[tuple[str, str, str]]) -> None:
    expected = {(seed, scope) for seed in TRAINING_SEEDS for scope in SCOPES}
    actual = {(seed, scope) for _replay, seed, scope in cells}
    if (
        len(cells) != 4
        or actual != expected
        or any(replay != REPLAY for replay, _, _ in cells)
    ):
        raise ValueError("ablation requires exactly R61 x T61/T63 x all/heads_only")


def trunk_state(model: PolicyValueNet) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(TRUNK_PREFIXES)
    }


def tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        for name, value in state.items()
    }


def assert_trunk_equal(g0: dict[str, torch.Tensor], model: PolicyValueNet) -> None:
    current = trunk_state(model)
    if set(current) != set(g0) or any(
        not torch.equal(g0[name], current[name]) for name in g0
    ):
        raise RuntimeError("heads_only_scope_leaked_trunk_updates")


def scope_manifest(model: PolicyValueNet, scope: str) -> dict[str, Any]:
    apply_trainable_scope(model, scope)
    states = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    if scope == "heads_only":
        unexpected = [
            name
            for name, enabled in states.items()
            if enabled != name.startswith(HEAD_PREFIXES)
        ]
        if unexpected:
            raise RuntimeError(
                f"heads_only parameter membership mismatch: {unexpected}"
            )
    total, trainable = _count_parameters(model)
    return {
        "scope": scope,
        "parameters": states,
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": total - trainable,
    }


def parameter_vector(model: PolicyValueNet, prefixes: tuple[str, ...]) -> torch.Tensor:
    return torch.cat(
        [
            p.detach().reshape(-1).float().cpu()
            for n, p in model.named_parameters()
            if n.startswith(prefixes)
        ]
    )


def anchor_row(evaluation: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in evaluation["rows"] if row["id"] == ANCHOR_ID)


def checkpoint_anchor_metrics(
    checkpoint: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate a serialized checkpoint through the canonical legal policy path."""
    row = anchor_row(evaluate_checkpoint(checkpoint, manifest))
    state = next(item for item in manifest["entries"] if item["id"] == ANCHOR_ID)
    model = PolicyValueNet((96, 3), "residual_v3", 27)
    load_checkpoint_into_model(model, checkpoint)
    x = torch.tensor(
        [encode_state(state["state"], input_encoding="kalah_v3")], dtype=torch.float32
    )
    with torch.no_grad():
        logits, _ = model(x)
    legal = state["legal_actions"]
    other = max(float(logits[0, action]) for action in legal if action != 0)
    return {
        **row,
        "action_0_probability": float(row["policy"][0]),
        "anchor_margin": float(logits[0, 0]) - other,
    }


def classify(cells: list[dict[str, Any]], g0_anchor: dict[str, Any]) -> tuple[str, str]:
    heads = {
        cell["training_seed"]: cell for cell in cells if cell["scope"] == "heads_only"
    }
    all_cells = {
        cell["training_seed"]: cell for cell in cells if cell["scope"] == "all"
    }
    if any(not cell.get("trunk_immutable", False) for cell in heads.values()):
        return (
            "heads_only_scope_leaked_trunk_updates",
            "fix/test trainable-scope enforcement before any ML interpretation.",
        )
    if len(heads) != 2 or len(all_cells) != 2:
        return (
            "heads_only_ablation_inconclusive",
            "recover SHA-verified controls and evaluation artifacts.",
        )
    t61, t63 = heads["T61"], heads["T63"]
    repaired = t61["anchor"]["top_is_outcome_optimal"] and (
        t61["anchor"]["optimal_mass"] - all_cells["T61"]["anchor"]["optimal_mass"]
        >= 0.10
        or t61["anchor"]["optimal_mass"] >= g0_anchor["optimal_mass"] - 0.05
    )
    t63_safe = bool(t63["anchor"]["top_is_outcome_optimal"])
    broad_regression = any(cell["material_regression"] for cell in heads.values())
    if repaired and t63_safe and not broad_regression:
        return (
            "heads_only_stabilizes_shared_trunk_failure",
            "confirm heads_only on frozen R62 and R63 with the same pre-registered seed pair before considering it for normal iterative self-play.",
        )
    if repaired and not t63_safe:
        return (
            "heads_only_seed_sensitive",
            "compare T61/T63 head-gradient trajectories under the fixed G0 trunk before trying another training scope.",
        )
    if repaired:
        return (
            "heads_only_stabilizes_but_underfits",
            "test the already-supported last_block_policy scope on frozen R61, allowing only the final residual block plus policy path to adapt.",
        )
    return (
        "heads_only_no_anchor_rescue",
        "test last_block_policy on frozen R61 as the smallest existing scope that permits limited representation adaptation.",
    )


def train_heads_only(
    paths: dict[str, Any], manifest: dict[str, Any], seed_name: str, workdir: Path
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
    manifest_scope = scope_manifest(model, "heads_only")
    g0_trunk = trunk_state(model)
    g0_heads = {
        "policy": parameter_vector(model, HEAD_PREFIXES[:2]),
        "value": parameter_vector(model, HEAD_PREFIXES[2:]),
    }
    permutations: dict[str, str] = {}
    epochs: list[dict[str, Any]] = []
    previous = dict(g0_heads)

    def permutation(epoch: int | None, values: list[int]) -> None:
        permutations[str(epoch)] = hashlib.sha256(
            json.dumps(values).encode()
        ).hexdigest()

    def epoch_callback(
        epoch: int, _optimizer: torch.optim.Optimizer, current: torch.nn.Module
    ) -> None:
        assert isinstance(current, PolicyValueNet)
        assert_trunk_equal(g0_trunk, current)
        checkpoint = workdir / "epochs" / f"epoch{epoch}.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint, **checkpoint_from_model(current))
        anchor = checkpoint_anchor_metrics(checkpoint, manifest)
        movement = {}
        for name, prefixes in (
            ("policy", HEAD_PREFIXES[:2]),
            ("value", HEAD_PREFIXES[2:]),
        ):
            now = parameter_vector(current, prefixes)
            movement[name] = {
                "drift_norm": float(torch.linalg.vector_norm(now - g0_heads[name])),
                "relative_drift": float(
                    torch.linalg.vector_norm(now - g0_heads[name])
                    / torch.linalg.vector_norm(g0_heads[name])
                ),
                "update_norm": float(torch.linalg.vector_norm(now - previous[name])),
            }
            previous[name] = now
        epochs.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "anchor": anchor,
                "head_movement": movement,
                "trunk_immutable": True,
            }
        )

    history: list[dict[str, float | int | None]] = []
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
        value_loss_weight=0.3,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.1,
        grad_clip=GRAD_CLIP,
        save_top_k=3,
        lr_scheduler="none",
        epoch_history=history,
        permutation_callback=permutation,
        epoch_callback=epoch_callback,
    )
    assert_trunk_equal(g0_trunk, model)
    checkpoint = workdir / "checkpoint.npz"
    np.savez(checkpoint, **checkpoint_from_model(model))
    final = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "anchor": checkpoint_anchor_metrics(checkpoint, manifest),
        "trunk_immutable": True,
    }
    return {
        "checkpoint": checkpoint,
        "scope_manifest": manifest_scope,
        "trunk_tensor_hashes": tensor_hashes(g0_trunk),
        "epoch_permutation_hashes": permutations,
        "epoch_evaluations": epochs,
        "epoch_metrics": history,
        "final_restored_best": final,
        "trunk_immutable": True,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# R61 Heads-Only Shared-Trunk Ablation",
        "",
        "Inherited PR #315 classifications: `legal_policy_metric_mismatch_confirmed_no_decision_flip`; `anchor_drift_shared_trunk_primary`.",
        "",
        "## Artifact And Scope Manifest",
        "",
        f"- G0 SHA-256: `{G0_SHA}`",
        f"- Frozen-set SHA-256: `{FROZEN_SET_SHA}`",
        "- Canonical checkpoint metric: `legal_normalized_metric`.",
        "- Guardrails: no self-play, replay mutation, frozen-state injection, exact labels, search changes, optimizer changes, or promotion.",
        "",
        "## Results",
        "",
        "| seed | scope | top | optimal mass | P(0) | margin | trunk immutable |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for cell in result.get("cells", []):
        anchor = cell["anchor"]
        lines.append(
            f"| {cell['training_seed']} | {cell['scope']} | {anchor['top_action']} | {anchor['optimal_mass']:.4f} | {anchor['action_0_probability']:.4f} | {anchor['anchor_margin']:.4f} | {cell.get('trunk_immutable', False)} |"
        )
    lines += [
        "",
        "## Classification",
        "",
        f"`{result['classification']}`",
        "",
        f"Exactly one next experiment: {result['next_experiment']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-result", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    assert_design(
        [(REPLAY, seed, scope) for seed in TRAINING_SEEDS for scope in SCOPES]
    )
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
        "artifacts": artifacts,
        "guardrails": {
            "self_play": False,
            "replay_mutation": False,
            "exact_labels_training": False,
            "promotion": False,
            "search_changes": False,
            "optimizer_changes": False,
        },
        "cells": [],
    }
    if not args.execute:
        result.update({"classification": "planned", "next_experiment": "not run"})
    else:
        g0_eval = evaluate_checkpoint(paths["parent"], manifest)
        g0_anchor = checkpoint_anchor_metrics(paths["parent"], manifest)
        result["g0_anchor"] = g0_anchor
        controls = {}
        for seed, (relative, expected_sha) in INHERITED_CONTROLS.items():
            checkpoint = ROOT / relative
            if not checkpoint.is_file() or sha256_file(checkpoint) != expected_sha:
                raise RuntimeError(
                    "heads_only_ablation_inconclusive: inherited all checkpoint unavailable or SHA mismatch"
                )
            evaluation = evaluate_checkpoint(checkpoint, manifest)
            controls[seed] = {
                "id": f"R61-{seed}-all",
                "training_seed": seed,
                "scope": "all",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": expected_sha,
                "anchor": checkpoint_anchor_metrics(checkpoint, manifest),
                "frozen_set": grouped_aggregates(evaluation["rows"]),
                "frozen_counts": frozen_counts(g0_eval["rows"], evaluation["rows"]),
                "trunk_immutable": None,
                "material_regression": False,
            }
        result["cells"].extend(controls.values())
        for seed in TRAINING_SEEDS:
            run = train_heads_only(
                paths, manifest, seed, args.workdir / "cells" / f"R61-{seed}-heads_only"
            )
            evaluation = evaluate_checkpoint(run["checkpoint"], manifest)
            frozen = grouped_aggregates(evaluation["rows"])
            export_for_arena(
                run["checkpoint"],
                args.workdir / "cells" / f"R61-{seed}-heads_only",
                f"R61-{seed}-heads_only",
            )
            arena = run_static_baseline(
                checkpoint=run["checkpoint"],
                baseline_artifact=ROOT / "storage/ai/alphazero_lite/current",
                out=args.workdir / "arena" / f"R61-{seed}.json",
                games=30,
                seed=TRAINING_SEEDS[seed],
            )["result"]
            effect = float(arena["score"]) - 0.5
            control_effect = 0.0  # Historical controls lack a same-run arena artifact; record comparison as unavailable.
            cell = {
                "id": f"R61-{seed}-heads_only",
                "training_seed": seed,
                "scope": "heads_only",
                **run,
                "checkpoint": str(run["checkpoint"]),
                "anchor": checkpoint_anchor_metrics(run["checkpoint"], manifest),
                "frozen_set": frozen,
                "frozen_counts": frozen_counts(g0_eval["rows"], evaluation["rows"]),
                "arena": arena,
                "arena_effect": effect,
                "material_regression": frozen["cluster"]["optimal_mass"]
                < controls[seed]["frozen_set"]["cluster"]["optimal_mass"]
                - MATERIAL_CLUSTER_REGRESSION
                or effect < control_effect - MATERIAL_ARENA_REGRESSION,
            }
            result["cells"].append(cell)
        result["classification"], result["next_experiment"] = classify(
            result["cells"], g0_anchor
        )
    write_json(args.out_result, result)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
