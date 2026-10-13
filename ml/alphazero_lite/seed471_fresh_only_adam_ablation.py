"""Seed471 sixteen-step fresh-only Adam learning-rate ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed471-fresh-only-adam-ablation")
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
INPUTS = {
    "manifest": "docs/data/seed466-replay-input-bundle/manifest.json",
    "training": "docs/data/seed466-replay-input-bundle/training.npz",
    "evaluation": "docs/data/seed466-replay-input-bundle/evaluation.npz",
    "initializer": str(INIT),
    "seed469_registration": "docs/data/seed469-optimizer-direction-correction/registration.json",
    "seed469_receipt": "docs/data/seed469-optimizer-direction-correction/receipt.json",
    "seed470_registration": "docs/data/seed470-finite-step-transport/registration.json",
    "seed470_receipt": "docs/data/seed470-finite-step-transport/receipt.json",
}
SOURCES = (
    "ml/alphazero_lite/seed471_fresh_only_adam_ablation.py",
    "ml/alphazero_lite/publish_seed471_fresh_only_adam_ablation.py",
    "ml/alphazero_lite/verify_seed471_fresh_only_adam_ablation.py",
    "ml/alphazero_lite/test_seed471_fresh_only_adam_ablation.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/seed466_runtime_bundle.py",
    "ml/alphazero_lite/seed466_replay_bundle.py",
    "ml/alphazero_lite/verify_seed466_replay_bundle.py",
    "ml/alphazero_lite/checkpoint_phase_selection.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/kalah_rules.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    if out.exists():
        if any(out.iterdir()):
            raise ValueError("seed471_output_exists")
        out.rmdir()
    bundle = root / "docs/data/seed466-replay-input-bundle"
    arrays = read_authenticated_bundle(bundle, MANIFEST_SHA256)
    tr, ev = arrays["training"], arrays["evaluation"]
    batches = tr["first16_minibatch_positions"][:8192]
    if len(batches) != 8192:
        raise ValueError("seed466_first16_positions_mismatch")
    fresh = np.flatnonzero(ev["source_ids"] == 0)
    pooled = (
        np.flatnonzero(ev["subset"] == 0)
        if "subset" in ev
        else np.arange(len(ev["source_ids"]))
    )
    # Bundle evaluation contains strict-unseen >32 occurrences; source_ids identifies fresh.
    if len(fresh) != 859 or len(pooled) != 2607:
        raise ValueError("authenticated_evaluation_cohort_mismatch")
    reg = {
        "schema": "seed471-fresh-only-adam-ablation-registration-v1",
        "status": "frozen_before_execution",
        "hypothesis": "Adam learning rate 0.0000625 avoids finite-step overshoot and improves fresh-unseen policy CE across sixteen sequential updates",
        "interpretation": "exploratory analysis informed by previously inspected results; existing occurrences are not a new holdout or independent confirmation",
        "inputs_sha256": {key: sha(root / path) for key, path in INPUTS.items()},
        "source_closure_sha256": {path: sha(root / path) for path in SOURCES},
        "authenticated_manifest_sha256": MANIFEST_SHA256,
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "architecture": {
            "hidden_sizes": [96, 3],
            "name": "residual_v3",
            "input_size": 27,
        },
        "minibatches": {
            "source": "authenticated seed466 first16_minibatch_positions",
            "batch_size": 512,
            "positions": batches.tolist(),
            "updates": 16,
        },
        "arms": {"C": {"lr": 0.001}, "T": {"lr": 0.0000625}},
        "optimizer": {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "gradient_clip_norm": 1.0,
            "initial_moments": "zero",
            "sequential_state_per_arm": True,
        },
        "objective": {
            "J": "sum(c_i*fresh_i*legal_CE_i)/sum(c_i*fresh_i) + 0.3*sum(fresh_i*Huber_i)/512",
            "huber_delta": 1.0,
            "value_denominator": 512,
            "historical_rows": "contribute neither policy nor value loss",
            "training_J_population": "all 134502 expanded occurrences at frozen train_positions, in stored order with multiplicity",
            "training_J_aggregation": "policy numerator sum(c*fresh*CE), policy denominator sum(c*fresh); value numerator sum(fresh*Huber), value denominator 134502; J=policy numerator/policy denominator+0.3*value numerator/value denominator",
        },
        "cohorts": {
            "fresh": "evaluation source_ids==0, preserve occurrence order and targets",
            "pooled": "all authenticated evaluation occurrences, preserve occurrence order and targets",
            "fresh_occurrences": len(fresh),
            "fresh_identities": len(set(ev["input_identity"][fresh].tolist())),
            "pooled_occurrences": len(pooled),
        },
        "weighting": {
            "exposure_weighted": "arithmetic mean over occurrences",
            "equal_input": "mean within exact input_identity groups, then mean across identities",
        },
        "numeric_semantics": {
            "training": "float32 torch objective, gradient, clip_grad_norm_, torch Adam step; parameters and moments remain float32",
            "evaluation": "float32 model inference and row errors; float64 weighted aggregate",
            "gradient_archive": "post-clipping gradients",
            "unused_parameters": "zero gradients; Adam state maintained for every parameter",
            "threads": 1,
        },
        "decision_rules": {
            "fresh_ce_improvement_vs_initializer": 0.001,
            "fresh_ce_improvement_vs_C": 0.001,
            "pooled_ce_max_regression_vs_initializer": 0.005,
            "pooled_mse_max_regression_vs_initializer": 0.01,
            "training_J": "T endpoint objective strictly below initializer",
            "decision_pass": "advance_to_fresh_confirmation",
            "decision_fail": "close_small_step_fresh_only_branch",
            "authorization": "separately designed fresh confirmation only; no arena execution or promotion",
        },
    }
    out.mkdir(parents=True)
    (out / "registration.json").write_text(
        json.dumps(reg, indent=2, sort_keys=True) + "\n"
    )
    return reg


def run_arm(root: Path, arm: str) -> None:
    root = root.resolve()
    out = root / OUT
    reg = json.loads((out / "registration.json").read_text())
    if reg["status"] != "frozen_before_execution" or arm not in reg["arms"]:
        raise ValueError("invalid_execution_state")
    if arm == "T" and not (out / "C-steps.npz").is_file():
        raise ValueError("C_must_run_before_T")
    for rel, expected in reg["source_closure_sha256"].items():
        if sha(root / rel) != expected:
            raise ValueError(f"source_changed:{rel}")
    for key, expected in reg["inputs_sha256"].items():
        rel = INPUTS[key]
        if sha(root / rel) != expected:
            raise ValueError(f"input_changed:{key}")
    arrays = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr = arrays["training"]
    # Each occurrence position addresses the frozen training-position vector, then replay expansion.
    compact = tr["expanded_replay_indices"][
        tr["train_positions"][
            np.asarray(reg["minibatches"]["positions"], dtype=np.int64)
        ]
    ]
    x, p = tr["inputs"][compact], tr["policy_targets"][compact]
    masks = tr["legal_masks"][compact]
    values = tr["value_targets"][compact].reshape(-1)
    coeff = tr["policy_coefficients"][compact]
    fresh = tr["fresh_source_mask"][compact].astype(np.float32)
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, root / INIT)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(
        params, lr=reg["arms"][arm]["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    archive: dict[str, np.ndarray] = {}
    steps = []
    torch.set_num_threads(1)
    for step in range(16):
        sl = slice(step * 512, (step + 1) * 512)
        xb, target, mask = (torch.from_numpy(v[sl].copy()) for v in (x, p, masks))
        c = torch.from_numpy(coeff[sl].copy())
        f = torch.from_numpy(fresh[sl].copy())
        value_target = torch.from_numpy(values[sl].copy())
        pre = [v.detach().clone() for v in params]
        optimizer.zero_grad(set_to_none=True)
        logits, prediction = model(xb)
        legal_logits = logits.masked_fill(mask == 0, -1e9)
        ce = -(target * F.log_softmax(legal_logits, dim=1)).sum(dim=1)
        weight = c * f
        mass = weight.sum()
        if float(mass) <= 0:
            raise ValueError(f"empty_fresh_policy_mass:{step}")
        policy = (weight * ce).sum() / mass
        huber = F.smooth_l1_loss(prediction.reshape(-1), value_target, reduction="none")
        value = (huber * f).sum() / 512
        objective = policy + 0.3 * value
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        grads = [v.grad.detach().clone() for v in params]
        optimizer.step()
        post = [v.detach().clone() for v in params]
        steps.append(
            {
                "step": step + 1,
                "objective": float(objective.detach()),
                "policy_loss": float(policy.detach()),
                "value_loss": float(value.detach()),
                "policy_mass": float(mass),
                "preclip_norm": float(norm),
                "clipped": bool(float(norm) > 1.0),
                "occurrence_indices": compact[sl].tolist(),
                "occurrence_identities": [
                    f"source={int(tr['source_ids'][row])};raw_line={int(tr['raw_lines'][row])};compact_row={int(row)}"
                    for row in compact[sl]
                ],
            }
        )
        for i, (a, g, b) in enumerate(zip(pre, grads, post, strict=True)):
            for label, tensor in (
                ("pre", a),
                ("gradient", g),
                ("post", b),
                ("exp_avg", optimizer.state[params[i]]["exp_avg"]),
                ("exp_avg_sq", optimizer.state[params[i]]["exp_avg_sq"]),
            ):
                archive[f"{step:02d}_{label}_{i:02d}"] = (
                    tensor.detach().cpu().numpy().copy()
                )
            archive[f"{step:02d}_displacement_{i:02d}"] = (
                (b - a).detach().cpu().numpy().copy()
            )
    np.savez_compressed(out / f"{arm}-steps.npz", **archive)
    (out / f"{arm}-trajectory.json").write_text(
        json.dumps({"arm": arm, "steps": steps}, indent=2, sort_keys=True) + "\n"
    )


def reset_unverified(root: Path) -> None:
    """Remove only a seed471 publication lacking an independent verification marker."""
    out = root.resolve() / OUT
    if not out.exists():
        return
    if (out / "verified.json").exists():
        raise ValueError("refusing_to_remove_verified_seed471_publication")
    registration = json.loads((out / "registration.json").read_text())
    if registration.get("schema") != "seed471-fresh-only-adam-ablation-registration-v1":
        raise ValueError("refusing_to_remove_unrecognized_seed471_output")
    if all(
        sha(root / relative) == expected
        for relative, expected in registration.get("source_closure_sha256", {}).items()
    ):
        raise ValueError("refusing_to_remove_currently_frozen_seed471_output")
    shutil.rmtree(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("freeze", "run-C", "run-T", "reset-unverified")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root)
    elif args.action == "reset-unverified":
        reset_unverified(args.root)
    else:
        run_arm(args.root, args.action[-1])


if __name__ == "__main__":
    main()
