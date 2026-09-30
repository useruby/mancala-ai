#!/usr/bin/env python3
"""Reproduce frozen generations while observing policy/value trunk gradients only."""

# ruff: noqa: E402

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

from ml.alphazero_lite.checkpoint_phase_selection import phase_mask_for_encoded_states  # noqa: E402
from ml.alphazero_lite.policy_value_gradient_audit import (
    gradient_metrics,
    gradients,
    parameter_groups,
    summarize,
)  # noqa: E402
from ml.alphazero_lite.train import (
    PolicyValueNet,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    train,
    weighted_policy_loss,
)  # noqa: E402

SCHEMA = "policy-value-shared-trunk-gradient-conflict-audit-v1"
VALUE_WEIGHT = 0.3
BATCH_SIZE = 512
EPOCHS = 4
GRAD_CLIP = 1.0
REPLAY_WEIGHTS = [1, 4, 1, 8, 4]
REPLAY_MODES = ["default", "sharpened", "sharpened", "sharpened", "sharpened"]
FIXED = [
    ROOT / ".tmp/fresh-uniform1200/replay-regeneration/generic_bootstrap.jsonl",
    ROOT / ".tmp/fresh-uniform1200/replay-regeneration/random_teacher_1200_train.jsonl",
    ROOT
    / ".tmp/fresh-uniform1200/replay-regeneration/opening-disagreement/opening_puct_disagreement_replay.jsonl",
    ROOT
    / ".tmp/fresh-uniform1200/replay-regeneration/stability/equal_budget_stability_replay.jsonl",
]
COHORTS = {
    "S455": {
        "generation": "seed48-nextgen-s455-default-value",
        "seed": 455,
        "fresh": ROOT
        / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/self_play.jsonl",
        "parent": ROOT
        / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/parent_init_checkpoint.npz",
        "epochs": {
            "E1": "7777c49291bce498ba23461c28e6c32799eb9ec14eb889003c17d4bf2c559e75",
            "E2": "3efd725e2f873b16572feaf79f27d81344a99ea4ead48052a8be3f8ba19d0734",
            "E3": "c18beeaa16ebaa038cd383cfd222705c636e2b6398c7270e6674d33231aa9dd1",
            "E4": "f531561bf303f0f14913f1cb47330823bff1e30e2abd11493d598327f0b22895",
        },
    },
    "F461": {
        "generation": "seed455-nextgen-s461-default-value-root16",
        "seed": 461,
        "fresh": ROOT
        / ".tmp/seed455-nextgen-s461-default-value-root16/runs/seed455-nextgen-s461-default-value-root16-iter1/self_play.jsonl",
        "parent": ROOT
        / ".tmp/seed455-nextgen-s461-default-value-root16/runs/seed455-nextgen-s461-default-value-root16-iter1/parent_init_checkpoint.npz",
        "epochs": {
            "E1": "643407f0b070acc1603e90e14e4ad8fd4579e4e34e3c109d9b0691f08267a1dc",
            "E2": "73914b801e61be7eada0366619fbc2defb23999fb163ef0da49134594ec44488",
            "E3": "9d3db177d22f12b0c8952233f3fc6c5dad5b0529f3990932699e8839afa0d2b8",
            "E4": "4bc05f8284d5adb5beac5090dbf2c09686c84831131cf3ceede606587ec127f4",
        },
    },
    "U467": {
        "generation": "seed467-unanchored-control",
        "seed": 467,
        "fresh": ROOT
        / ".tmp/seed455-nextgen-s467-parent-anchor-w010/runs/seed455-nextgen-s467-parent-anchor-w010-iter1/self_play.jsonl",
        "parent": ROOT
        / ".tmp/seed467-unanchored-control/runs/seed467-unanchored-control-iter1/parent_init_checkpoint.npz",
        "epochs": {
            "E1": "0ea38854e5610997080a419f6e6db1a181967a130eb5494e8ec31676c18ac3fe",
            "E2": "584b4861c818fcbdcd17ee0f95beb572b4db4a5eb15384149711f66fd608f7f3",
            "E3": "cc4534f980507d2332ef1cee59ce3abcdd860ac2513b010aa1305846e7d28360",
            "E4": "798b4654a5e16ceb4dd675193074b8150da83e1887d7c78202c1890d2ff71f2f",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sha(model: PolicyValueNet, path: Path) -> str:
    from ml.alphazero_lite.train import checkpoint_from_model

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **checkpoint_from_model(model))
    return sha256(path)


def source_lookup(paths: list[Path]) -> np.ndarray:
    """Map compact row positions to fresh or historical without changing replay."""
    counts = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            counts.append(sum(1 for _ in handle))
    return np.concatenate(
        [
            np.full(count, "fresh" if index == 0 else "historical")
            for index, count in enumerate(counts)
        ]
    )


def aggregate_by_epoch(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        f"E{epoch}": summarize([row[key] for row in rows if row["epoch"] == epoch])
        for epoch in range(1, EPOCHS + 1)
    }


def classify(result: dict[str, Any]) -> str:
    # The labels intentionally use only the preregistered generation-level criteria.
    summaries = {name: data["epoch_high"] for name, data in result["cohorts"].items()}
    control = summaries["S455"]
    failures = [summaries["F461"], summaries["U467"]]
    more_conflict = all(
        sum(
            item[f"E{epoch}"]["conflict_fraction"]
            > control[f"E{epoch}"]["conflict_fraction"]
            for epoch in range(1, 5)
        )
        >= 3
        for item in failures
    )
    dominance = all(
        statistics.fmean(
            item[f"E{epoch}"].get("mean_value_policy_norm_ratio", 0.0)
            for epoch in range(1, 5)
        )
        > 2.0
        * statistics.fmean(
            control[f"E{epoch}"].get("mean_value_policy_norm_ratio", 0.0)
            for epoch in range(1, 5)
        )
        for item in failures
    )
    common = all(
        abs(
            statistics.fmean(
                item[f"E{epoch}"]["conflict_fraction"] for epoch in range(1, 5)
            )
            - statistics.fmean(
                control[f"E{epoch}"]["conflict_fraction"] for epoch in range(1, 5)
            )
        )
        < 0.03
        for item in failures
    )
    if common:
        return "shared_trunk_conflict_common_to_all_generations"
    if more_conflict:
        return "shared_trunk_policy_value_conflict_failure_signature"
    if dominance:
        return "shared_trunk_value_gradient_dominance_failure_signature"
    return "policy_value_gradient_conflict_no_clear_signal"


def run_cohort(name: str, spec: dict[str, Any], workdir: Path) -> dict[str, Any]:
    paths = [spec["fresh"], *FIXED]
    for path in [*paths, spec["parent"]]:
        if not path.exists():
            raise RuntimeError(f"gradient_audit_input_missing:{path}")
    set_seed(spec["seed"])
    loaded = load_jsonl_replay(
        paths,
        REPLAY_WEIGHTS,
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=REPLAY_MODES,
        include_policy_loss_weights=True,
    )
    x, p, v, replay_indexes, policy_weights = loaded
    model = PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    load_checkpoint_into_model(model, spec["parent"])
    groups = parameter_groups(model)
    trunk = tuple(
        parameter
        for key in (
            "input_projection",
            "residual_block_0",
            "residual_block_1",
            "residual_block_2",
        )
        for parameter in groups[key]
    )
    high_mask = phase_mask_for_encoded_states(x)
    sources = source_lookup(paths)
    traces: list[dict[str, Any]] = []
    epochs: dict[str, str] = {}
    permutations: dict[str, str] = {}

    def observe(context: dict[str, Any]) -> None:
        if (
            context["value_loss_weight"] != VALUE_WEIGHT
            or context["pairwise_loss_weight"] != 0.0
            or context["behavior_loss_weight"] != 0.0
        ):
            raise RuntimeError("gradient_audit_training_contract_mismatch")
        # Later minibatches intentionally still hold the preceding .grad until
        # production calls zero_grad after this observer. Preserve it bytewise.
        prior_grads = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        policy = gradients(context["policy_loss_tensor"], trunk)
        raw_value = gradients(context["value_loss_tensor"], trunk)
        weighted_value = tuple(VALUE_WEIGHT * item for item in raw_value)
        batch = np.asarray(context["batch_indexes"], dtype=np.int64)
        row: dict[str, Any] = {
            "epoch": int(context["epoch"]),
            "all": gradient_metrics(policy, weighted_value, raw_value),
            "high_rows": int(high_mask[batch].sum()),
            "source": str(sources[batch[0]])
            if len(set(sources[batch])) == 1
            else "mixed",
        }

        def slice_metrics(
            selected: np.ndarray, *, blocks: bool = False
        ) -> dict[str, Any] | None:
            if not selected.size:
                return None
            # Reuse the production loss primitives and legal masks on the exact batch subset.
            bx = torch.from_numpy(x[selected])
            bp, bv = torch.from_numpy(p[selected]), torch.from_numpy(v[selected])
            bw = torch.from_numpy(policy_weights[selected])
            mask = torch.from_numpy(legal_mask_matrix_for_encoded_states(x[selected]))
            logits, prediction = model(bx)
            high_policy_loss = weighted_policy_loss(
                compute_policy_cross_entropy(logits.masked_fill(mask <= 0.0, -1e9), bp),
                bw,
            )
            high_value_loss = compute_value_loss_vector(
                prediction, bv, value_loss="huber", huber_delta=1.0
            ).mean()
            hp = gradients(high_policy_loss, trunk)
            hraw = gradients(high_value_loss, trunk)
            hw = tuple(VALUE_WEIGHT * item for item in hraw)
            result: dict[str, Any] = {"metrics": gradient_metrics(hp, hw, hraw)}
            if blocks:
                result["blocks"] = {}
                for key in (
                    "input_projection",
                    "residual_block_0",
                    "residual_block_1",
                    "residual_block_2",
                ):
                    block_policy = gradients(high_policy_loss, groups[key])
                    block_raw = gradients(high_value_loss, groups[key])
                    result["blocks"][key] = gradient_metrics(
                        block_policy,
                        tuple(VALUE_WEIGHT * item for item in block_raw),
                        block_raw,
                    )
            return result

        high = slice_metrics(batch[high_mask[batch]], blocks=True)
        row["high"] = None if high is None else high["metrics"]
        row["blocks"] = {} if high is None else high["blocks"]
        row["source_slices"] = {}
        for source in ("fresh", "historical"):
            source_batch = batch[sources[batch] == source]
            source_high = source_batch[high_mask[source_batch]]
            all_slice = slice_metrics(source_batch)
            high_slice = slice_metrics(source_high)
            row["source_slices"][source] = {
                "all": None if all_slice is None else all_slice["metrics"],
                "high": None if high_slice is None else high_slice["metrics"],
            }
        if any(
            (prior_grads[name] is None) != (parameter.grad is None)
            or (
                prior_grads[name] is not None
                and not torch.equal(prior_grads[name], parameter.grad)
            )
            for name, parameter in model.named_parameters()
        ):
            raise RuntimeError("gradient_observer_changes_training")
        traces.append(row)

    def step_callback(phase: str, context: dict[str, Any]) -> None:
        if phase == "before":
            traces[-1]["clip_active"] = bool(context["clip_active"])
            traces[-1]["clip_scale"] = float(context["clip_scale"])
            traces[-1]["preclip_total_norm"] = float(context["gradient_norm"])

    def epoch_callback(
        epoch: int, _optimizer: torch.optim.Optimizer, current: torch.nn.Module
    ) -> None:
        epochs[f"E{epoch}"] = checkpoint_sha(current, workdir / name / f"E{epoch}.npz")

    def permutation(epoch: int | None, values: list[int]) -> None:
        permutations[str(epoch)] = hashlib.sha256(
            json.dumps(values).encode()
        ).hexdigest()

    train(
        model,
        x,
        p,
        v,
        replay_indexes,
        policy_loss_weights=policy_weights,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=0.001,
        device=torch.device("cpu"),
        value_loss_weight=VALUE_WEIGHT,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.1,
        grad_clip=GRAD_CLIP,
        save_top_k=3,
        lr_scheduler="none",
        final_checkpoint="best_validation",
        loss_observer=observe,
        step_callback=step_callback,
        step_callback_needs_raw_gradients=False,
        epoch_callback=epoch_callback,
        permutation_callback=permutation,
    )
    expected = spec["epochs"]
    if expected and epochs != expected:
        raise RuntimeError(
            f"gradient_audit_training_reproduction_drift:{name}:{epochs}"
        )
    return {
        "generation": spec["generation"],
        "fresh_sha256": sha256(spec["fresh"]),
        "seed": spec["seed"],
        "parent_sha256": sha256(spec["parent"]),
        "epoch_checkpoint_sha256": epochs,
        "reproduction": "exact",
        "permutation_sha256": permutations,
        "steps": len(traces),
        "epoch_all": aggregate_by_epoch(traces, "all"),
        "epoch_high": aggregate_by_epoch(
            [row for row in traces if row["high"] is not None], "high"
        ),
        "source_all": {
            source: summarize(
                [
                    row["source_slices"][source]["all"]
                    for row in traces
                    if row["source_slices"][source]["all"] is not None
                ]
            )
            for source in ("fresh", "historical")
        },
        "source_high": {
            source: summarize(
                [
                    row["source_slices"][source]["high"]
                    for row in traces
                    if row["source_slices"][source]["high"] is not None
                ]
            )
            for source in ("fresh", "historical")
        },
        "block_high": {
            group: summarize(
                [row["blocks"][group] for row in traces if group in row["blocks"]]
            )
            for group in (
                "input_projection",
                "residual_block_0",
                "residual_block_1",
                "residual_block_2",
            )
        },
        "clipping": {
            "conflict_clip_rate": statistics.fmean(
                row["clip_active"] for row in traces if row["all"]["conflict"]
            ),
            "non_conflict_clip_rate": statistics.fmean(
                row["clip_active"] for row in traces if not row["all"]["conflict"]
            ),
            "clipped_median_cosine": float(
                np.median(
                    [
                        row["all"]["cosine"]
                        for row in traces
                        if row["clip_active"] and row["all"]["cosine"] is not None
                    ]
                )
            ),
            "unclipped_median_cosine": None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "semantic_identity": SCHEMA,
        "guardrails": {
            "candidate_training": 0,
            "self_play_games": 0,
            "arena_games": 0,
            "canonical_games": 0,
            "promotions": 0,
        },
        "cohorts": {},
    }
    for name, spec in COHORTS.items():
        result["cohorts"][name] = run_cohort(name, spec, args.workdir)
    result["classification"] = classify(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
