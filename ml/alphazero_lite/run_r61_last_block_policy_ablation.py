#!/usr/bin/env python3
"""Frozen-R61 last_block_policy trainable-scope ablation.

This train-only runner reuses R61 replay and the SHA-verified PR #315/#316
controls. It never generates self-play, mutates replay, changes search, or
promotes a checkpoint.
"""

from __future__ import annotations

import argparse
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
from ml.alphazero_lite.forensic_exact_references import outcome_regret
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

SCHEMA = "azlite_r61_last_block_policy_ablation_v1"
TRAINING_SEEDS = {"T61": 61, "T63": 63}
SCOPES = ("all", "heads_only", "last_block_policy")
EPOCHS, BATCH_SIZE, LR, GRAD_CLIP = 4, 512, 0.001, 1.0
MATERIAL_CLUSTER_REGRESSION = 0.02
MATERIAL_ARENA_REGRESSION = 0.15
INHERITED_FULL = {
    "T61": (
        ".tmp/r61-parameter-subspace-drift-audit/T61/checkpoint.npz",
        "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    ),
    "T63": (
        ".tmp/r61-parameter-subspace-drift-audit/T63/checkpoint.npz",
        "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
    ),
}
INHERITED_HEADS = {
    "T61": (
        ".tmp/r61-heads-only-ablation/cells/R61-T61-heads_only/checkpoint.npz",
        "684d4ab7c0e244534157af09dffa8a12749815e5015dd2eb492474be958692ba",
    ),
    "T63": (
        ".tmp/r61-heads-only-ablation/cells/R61-T63-heads_only/checkpoint.npz",
        "707a0e06cae9ab5eee8db7fef09d337e1e5e20b47bb84bcecf2403ba144dcf1e",
    ),
}
FROZEN_PREFIXES = ("input_layer.", "value_hidden_layer.", "value_head.")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify_inherited_checkpoints(
    checkpoints: dict[str, tuple[str, str]], scope: str
) -> dict[str, Path]:
    verified = {}
    for seed, (relative, expected_sha) in checkpoints.items():
        checkpoint = ROOT / relative
        if not checkpoint.is_file() or sha256_file(checkpoint) != expected_sha:
            raise RuntimeError(
                "last_block_policy_ablation_inconclusive: "
                f"inherited {scope} checkpoint unavailable or SHA mismatch"
            )
        verified[seed] = checkpoint
    return verified


def assert_design(cells: list[tuple[str, str, str]]) -> None:
    expected = {(seed, scope) for seed in TRAINING_SEEDS for scope in SCOPES}
    actual = {(seed, scope) for replay, seed, scope in cells if replay == REPLAY}
    if len(cells) != 6 or actual != expected:
        raise ValueError(
            "ablation requires exactly R61 x T61/T63 x all/heads_only/last_block_policy"
        )


def final_block_prefix(model: PolicyValueNet) -> str:
    count = len(model.residual_layers)
    if count != 3:
        raise RuntimeError(
            f"last_block_policy requires exactly three residual blocks, got {count}"
        )
    return f"residual_layers.{count - 1}."


def frozen_prefixes(model: PolicyValueNet) -> tuple[str, ...]:
    final_block_prefix(model)
    return (
        *FROZEN_PREFIXES,
        *(
            f"residual_layers.{index}."
            for index in range(len(model.residual_layers) - 1)
        ),
    )


def trainable_prefixes(model: PolicyValueNet) -> tuple[str, ...]:
    return (final_block_prefix(model), "policy_hidden_layer.", "policy_head.")


def scope_manifest(model: PolicyValueNet) -> dict[str, Any]:
    apply_trainable_scope(model, "last_block_policy")
    frozen, trainable = frozen_prefixes(model), trainable_prefixes(model)
    parameters = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    bad = [
        name
        for name, enabled in parameters.items()
        if enabled != name.startswith(trainable)
    ]
    if bad or not all(
        any(name.startswith(prefix) for name in parameters)
        for prefix in (*frozen, *trainable)
    ):
        raise RuntimeError(f"last_block_policy parameter membership mismatch: {bad}")
    total, enabled = _count_parameters(model)
    return {
        "scope": "last_block_policy",
        "final_block_prefix": final_block_prefix(model),
        "parameters": parameters,
        "total_parameter_count": total,
        "trainable_parameter_count": enabled,
        "frozen_parameter_count": total - enabled,
    }


def parameter_state(
    model: PolicyValueNet, prefixes: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if name.startswith(prefixes)
    }


def parameter_vector(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([state[name].reshape(-1).float() for name in sorted(state)])


def tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        for name, value in state.items()
    }


def assert_frozen_equal(before: dict[str, torch.Tensor], model: PolicyValueNet) -> None:
    current = parameter_state(model, frozen_prefixes(model))
    if set(before) != set(current) or any(
        not torch.equal(before[name], current[name]) for name in before
    ):
        raise RuntimeError("last_block_policy_scope_leak")


def anchor_metrics(checkpoint: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    row = next(
        row
        for row in evaluate_checkpoint(checkpoint, manifest)["rows"]
        if row["id"] == ANCHOR_ID
    )
    entry = next(entry for entry in manifest["entries"] if entry["id"] == ANCHOR_ID)
    model = PolicyValueNet((96, 3), "residual_v3", 27)
    load_checkpoint_into_model(model, checkpoint)
    with torch.no_grad():
        logits, _ = model(
            torch.tensor(
                [encode_state(entry["state"], input_encoding="kalah_v3")],
                dtype=torch.float32,
            )
        )
    legal = entry["legal_actions"]
    return {
        **row,
        "action_0_probability": float(row["policy"][0]),
        "anchor_margin": float(logits[0, 0])
        - max(float(logits[0, action]) for action in legal if action != 0),
    }


def movement(
    current: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    prior: dict[str, torch.Tensor],
) -> dict[str, float]:
    now, start, previous = (
        parameter_vector(current),
        parameter_vector(baseline),
        parameter_vector(prior),
    )
    drift = torch.linalg.vector_norm(now - start)
    return {
        "drift_norm": float(drift),
        "relative_drift": float(
            drift / max(torch.linalg.vector_norm(start), torch.tensor(1e-12))
        ),
        "update_norm": float(torch.linalg.vector_norm(now - previous)),
    }


def drift_cosines(
    g0: Path, full: dict[str, Path], candidates: dict[str, Path]
) -> dict[str, Any]:
    def block(path: Path) -> torch.Tensor:
        model = PolicyValueNet((96, 3), "residual_v3", 27)
        load_checkpoint_into_model(model, path)
        return parameter_vector(parameter_state(model, (final_block_prefix(model),)))

    start = block(g0)
    vectors = {
        f"full{seed[1:]}": block(path) - start for seed, path in full.items()
    } | {
        f"lastblock{seed[1:]}": block(path) - start for seed, path in candidates.items()
    }
    output: dict[str, Any] = {}
    for candidate in ("lastblock61", "lastblock63"):
        output[candidate] = {}
        for control in ("full61", "full63"):
            denominator = torch.linalg.vector_norm(
                vectors[candidate]
            ) * torch.linalg.vector_norm(vectors[control])
            output[candidate][control] = {
                "cosine": None
                if not denominator
                else float(
                    torch.dot(vectors[candidate], vectors[control]) / denominator
                ),
                "norm_ratio": float(
                    torch.linalg.vector_norm(vectors[candidate])
                    / max(
                        torch.linalg.vector_norm(vectors[control]), torch.tensor(1e-12)
                    )
                ),
            }
    return output


def exact_forensic(checkpoint: Path) -> dict[str, Any]:
    rows = json.loads(
        (
            ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json"
        ).read_text()
    )["rows"]
    model = PolicyValueNet((96, 3), "residual_v3", 27)
    load_checkpoint_into_model(model, checkpoint)
    evaluated = []
    for row in rows:
        if row["exact_status"] != "exact_solved":
            continue
        with torch.no_grad():
            logits, value = model(
                torch.tensor(
                    [encode_state(row["state"], input_encoding="kalah_v3")],
                    dtype=torch.float32,
                )
            )
        selected = max(row["legal_moves"], key=lambda action: float(logits[0, action]))
        regret = outcome_regret(row, selected)
        evaluated.append(
            (row, regret, abs(float(value[0]) - float(row["exact_root_value"])))
        )

    def summary(
        items: list[tuple[dict[str, Any], int | None, float]],
    ) -> dict[str, Any]:
        regrets = [int(regret) for _, regret, _ in items if regret is not None]
        if len(regrets) != len(items):
            raise RuntimeError("exact forensic row had no outcome regret")
        return {
            "roots": len(items),
            "outcome_optimal_accuracy": statistics.fmean(
                float(regret == 0) for regret in regrets
            ),
            "true_outcome_regret": statistics.fmean(regrets),
            "true_outcome_blunder_rate": statistics.fmean(
                float(regret > 0) for regret in regrets
            ),
            "value_mae": statistics.fmean(error for _, _, error in items),
        }

    return {
        "overall": summary(evaluated),
        **{
            bucket: summary(
                [item for item in evaluated if bucket in item[0].get("id", "")]
            )
            for bucket in ("capture_available", "sparse_endgame")
        },
    }


def train_last_block(
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
    manifest_scope = scope_manifest(model)
    frozen, baseline = (
        parameter_state(model, frozen_prefixes(model)),
        parameter_state(model, trainable_prefixes(model)),
    )
    prior, epochs, permutations = dict(baseline), [], {}

    def permutation(epoch: int | None, values: list[int]) -> None:
        permutations[str(epoch)] = hashlib.sha256(
            json.dumps(values).encode()
        ).hexdigest()

    def callback(
        epoch: int, _optimizer: torch.optim.Optimizer, current: torch.nn.Module
    ) -> None:
        assert isinstance(current, PolicyValueNet)
        assert_frozen_equal(frozen, current)
        checkpoint = workdir / "epochs" / f"epoch{epoch}.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint, **checkpoint_from_model(current))
        now = parameter_state(current, trainable_prefixes(current))
        epochs.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "anchor": anchor_metrics(checkpoint, manifest),
                "movement": movement(now, baseline, prior),
                "frozen_parameter_invariant": True,
            }
        )
        prior.clear()
        prior.update(now)

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
        epoch_callback=callback,
    )
    assert_frozen_equal(frozen, model)
    checkpoint = workdir / "checkpoint.npz"
    np.savez(checkpoint, **checkpoint_from_model(model))
    final = parameter_state(model, trainable_prefixes(model))
    return {
        "checkpoint": checkpoint,
        "scope_manifest": manifest_scope,
        "frozen_tensor_hashes": tensor_hashes(frozen),
        "frozen_parameter_invariant": True,
        "epoch_permutation_hashes": permutations,
        "epoch_evaluations": epochs,
        "epoch_metrics": history,
        "final_movement": movement(final, baseline, prior),
        "final_restored_best": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "anchor": anchor_metrics(checkpoint, manifest),
            "frozen_parameter_invariant": True,
        },
    }


def classify(cells: list[dict[str, Any]]) -> tuple[str, str]:
    lanes = {
        cell["training_seed"]: cell
        for cell in cells
        if cell["scope"] == "last_block_policy"
    }
    full = {cell["training_seed"]: cell for cell in cells if cell["scope"] == "all"}
    if len(lanes) != 2 or len(full) != 2:
        return (
            "last_block_policy_ablation_inconclusive",
            "recover SHA-verified inherited artifacts and evaluation results.",
        )
    if any(not cell["frozen_parameter_invariant"] for cell in lanes.values()):
        return (
            "last_block_policy_scope_leak",
            "fix/trainable-scope enforcement and rerun this same experiment.",
        )
    t61, t63 = lanes["T61"], lanes["T63"]
    rescue = (
        t61["anchor"]["top_is_outcome_optimal"]
        or t61["anchor"]["optimal_mass"] - full["T61"]["anchor"]["optimal_mass"] >= 0.10
    )
    t63_safe = t63["anchor"]["top_is_outcome_optimal"]
    broad = any(
        cell["material_regression"]
        or cell["critical_forensic_regression"]
        or cell["arena_material_regression"]
        for cell in lanes.values()
    )
    if rescue and t63_safe and any(cell["value_limited"] for cell in lanes.values()):
        return (
            "last_block_policy_stabilizes_but_value_limited",
            "test an existing scope or minimal scope extension that trains the final residual block + policy path + value head while keeping earlier trunk blocks frozen.",
        )
    if rescue and t63_safe and not broad:
        return (
            "last_block_policy_stabilizes_shared_trunk_failure",
            "confirm last_block_policy on frozen R62 and R63 with the same pre-registered T61/T63 training-seed pair before any normal self-play adoption.",
        )
    if rescue and not t63_safe:
        return (
            "last_block_policy_seed_sensitive",
            "audit T61/T63 final-block gradient-source decomposition into policy-loss versus value-loss components before testing another scope.",
        )
    if not rescue:
        return (
            "last_block_policy_no_anchor_rescue",
            "run a policy-vs-value gradient attribution audit on the shared trunk under the original full R61 T61/T63 runs, using the existing full-training trajectories.",
        )
    return (
        "last_block_policy_broad_regression",
        "run policy-vs-value gradient attribution on the full shared trunk before further scope restriction.",
    )


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# R61 Last-Block Policy Ablation",
        "",
        "Inherited #315: `anchor_drift_shared_trunk_primary`. Inherited #316: `heads_only_no_anchor_rescue`.",
        "",
        "## Classification",
        "",
        f"`{result['classification']}`",
        "",
        f"Exactly one next experiment: {result['next_experiment']}",
        "",
        "## Artifacts And Invariants",
        "",
        f"- G0 SHA-256: `{G0_SHA}`",
        f"- Frozen set SHA-256: `{FROZEN_SET_SHA}`",
        "- Canonical anchor metric: `legal_normalized_metric`.",
        "- Guardrails: no self-play, replay mutation, replay-weight, optimizer, LR, batch, epoch, clipping, architecture, target, search, or promotion change.",
        "",
        "| seed | scope | top | optimal mass | P(0) | margin | frozen invariant |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for cell in result.get("cells", []):
        anchor = cell["anchor"]
        lines.append(
            f"| {cell['training_seed']} | {cell['scope']} | {anchor['top_action']} | {anchor['optimal_mass']:.4f} | {anchor['action_0_probability']:.4f} | {anchor['anchor_margin']:.4f} | {cell.get('frozen_parameter_invariant')} |"
        )
    lines.extend(
        [
            "",
            "Machine-readable artifact contains the exact parameter manifest, per-epoch frozen hashes/movement/permutations/anchor trajectories, drift direction, frozen family, value, exact-forensic, and arena results.",
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
            "replay_weight_change": False,
            "optimizer_change": False,
            "promotion": False,
        },
        "cells": [],
    }
    if not args.execute:
        result.update({"classification": "planned", "next_experiment": "not run"})
    else:
        g0_eval = evaluate_checkpoint(paths["parent"], manifest)
        inherited: dict[tuple[str, str], dict[str, Any]] = {}
        for scope, checkpoints in (
            ("all", INHERITED_FULL),
            ("heads_only", INHERITED_HEADS),
        ):
            verified = verify_inherited_checkpoints(checkpoints, scope)
            for seed, checkpoint in verified.items():
                expected = checkpoints[seed][1]
                evaluation = evaluate_checkpoint(checkpoint, manifest)
                inherited[(seed, scope)] = {
                    "id": f"R61-{seed}-{scope}",
                    "training_seed": seed,
                    "scope": scope,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": expected,
                    "anchor": anchor_metrics(checkpoint, manifest),
                    "frozen_set": grouped_aggregates(evaluation["rows"]),
                    "frozen_counts": frozen_counts(g0_eval["rows"], evaluation["rows"]),
                    "frozen_parameter_invariant": None,
                }
        result["cells"].extend(inherited.values())
        candidates: dict[str, Path] = {}
        for seed in TRAINING_SEEDS:
            run = train_last_block(
                paths,
                manifest,
                seed,
                args.workdir / "cells" / f"R61-{seed}-last_block_policy",
            )
            checkpoint = run["checkpoint"]
            candidates[seed] = checkpoint
            evaluation, forensic = (
                evaluate_checkpoint(checkpoint, manifest),
                exact_forensic(checkpoint),
            )
            export_for_arena(
                checkpoint, checkpoint.parent, f"R61-{seed}-last_block_policy"
            )
            arena = run_static_baseline(
                checkpoint=checkpoint,
                baseline_artifact=ROOT / "storage/ai/alphazero_lite/current",
                out=args.workdir / "arena" / f"R61-{seed}.json",
                games=30,
                seed=TRAINING_SEEDS[seed],
            )["result"]
            full = inherited[(seed, "all")]
            frozen = grouped_aggregates(evaluation["rows"])
            effect = float(arena["score"]) - 0.5
            result["cells"].append(
                {
                    "id": f"R61-{seed}-last_block_policy",
                    "training_seed": seed,
                    "scope": "last_block_policy",
                    **run,
                    "checkpoint": str(checkpoint),
                    "anchor": run["final_restored_best"]["anchor"],
                    "frozen_set": frozen,
                    "frozen_counts": frozen_counts(g0_eval["rows"], evaluation["rows"]),
                    "exact_forensic": forensic,
                    "arena": arena,
                    "arena_effect": effect,
                    "arena_effect_confidence_interval_95": {
                        "lower": float(arena["confidence_interval_95"]["lower"]) - 0.5,
                        "upper": float(arena["confidence_interval_95"]["upper"]) - 0.5,
                    },
                    "material_regression": frozen["cluster"]["optimal_mass"]
                    < full["frozen_set"]["cluster"]["optimal_mass"]
                    - MATERIAL_CLUSTER_REGRESSION,
                    "arena_material_regression": effect < -MATERIAL_ARENA_REGRESSION,
                    "critical_forensic_regression": forensic["overall"][
                        "outcome_optimal_accuracy"
                    ]
                    < exact_forensic(Path(full["checkpoint"]))["overall"][
                        "outcome_optimal_accuracy"
                    ],
                    "value_limited": frozen["cluster"]["value_mae"]
                    > full["frozen_set"]["cluster"]["value_mae"],
                }
            )
        result["drift_direction"] = drift_cosines(
            paths["parent"],
            {
                seed: Path(cell["checkpoint"])
                for (seed, scope), cell in inherited.items()
                if scope == "all"
            },
            candidates,
        )
        result["secondary_vs_heads_only"] = {
            seed: {
                "anchor_mass_better": next(
                    cell
                    for cell in result["cells"]
                    if cell["training_seed"] == seed
                    and cell["scope"] == "last_block_policy"
                )["anchor"]["optimal_mass"]
                > inherited[(seed, "heads_only")]["anchor"]["optimal_mass"],
                "cluster_mass_better": next(
                    cell
                    for cell in result["cells"]
                    if cell["training_seed"] == seed
                    and cell["scope"] == "last_block_policy"
                )["frozen_set"]["cluster"]["optimal_mass"]
                > inherited[(seed, "heads_only")]["frozen_set"]["cluster"][
                    "optimal_mass"
                ],
            }
            for seed in TRAINING_SEEDS
        }
        result["classification"], result["next_experiment"] = classify(result["cells"])
    write_json(args.out_result, result)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
