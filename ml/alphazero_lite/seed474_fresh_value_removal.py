"""Seed474 fresh-value removal under per-update Adam reset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from ml.alphazero_lite.seed471_fresh_only_adam_ablation import INIT, INPUTS
from ml.alphazero_lite.seed473_adam_moment_reset import (
    OUT as SEED473_OUT,
    SOURCES as SEED473_SOURCES,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed474-fresh-value-removal")
SOURCES = (
    "ml/alphazero_lite/seed474_fresh_value_removal.py",
    "ml/alphazero_lite/publish_seed474_fresh_value_removal.py",
    "ml/alphazero_lite/verify_seed474_fresh_value_removal.py",
    "ml/alphazero_lite/test_seed474_fresh_value_removal.py",
    *SEED473_SOURCES,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_control(expected_path: Path, actual_path: Path) -> None:
    """Require exact post-step equality with all historical seed473 C states."""
    with (
        np.load(expected_path, allow_pickle=False) as expected,
        np.load(actual_path, allow_pickle=False) as actual,
    ):
        count = sum(key.startswith("00_post_") for key in expected.files)
        if set(expected.files) != set(actual.files):
            raise ValueError("control_archive_inventory_mismatch")
        for step in range(16):
            for index in range(count):
                key = f"{step:02d}_post_{index:02d}"
                if not np.array_equal(expected[key], actual[key]):
                    raise ValueError(
                        f"control_reproduction_mismatch:{step + 1}:{index}"
                    )


def freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    if out.exists() and any(out.iterdir()):
        raise ValueError("seed474_output_exists")
    arrays = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = arrays["training"], arrays["evaluation"]
    positions = tr["first16_minibatch_positions"][:8192]
    if len(positions) != 8192 or len(ev["source_ids"]) != 2607:
        raise ValueError("authenticated_schedule_or_cohort_mismatch")
    reg = {
        "schema": "seed474-fresh-value-removal-registration-v1",
        "status": "frozen_before_execution",
        "interpretation": "exploratory reused cohort; no independent-confirmation or strength claim",
        "hypothesis": "the remaining reset-Adam fresh-policy regression comes from the fresh-value term",
        "inputs_sha256": {key: sha(root / rel) for key, rel in INPUTS.items()},
        "source_closure_sha256": {rel: sha(root / rel) for rel in SOURCES},
        "authenticated_manifest_sha256": MANIFEST_SHA256,
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "initializer_sha256": sha(root / INIT),
        "seed473_R_steps_sha256": sha(root / SEED473_OUT / "R-steps.npz"),
        "seed473_validity_gate_sha256": sha(root / SEED473_OUT / "validity-gate.json"),
        "minibatches": {
            "positions": positions.tolist(),
            "updates": 16,
            "batch_size": 512,
            "source": "authenticated seed466 first16_minibatch_positions",
        },
        "architecture": {
            "hidden_sizes": [96, 3],
            "name": "residual_v3",
            "input_size": 27,
        },
        "arms": {
            "C": "seed473 objective reproduced exactly",
            "P": "fresh-policy CE only; value coefficient exactly zero",
        },
        "optimizer": {
            "name": "Adam",
            "lr": 0.0000625,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "gradient_clip_norm": 1.0,
            "reset_moments_and_counter_before_each_update": True,
            "unused_parameter_gradients": "explicit zero tensors for every parameter; optimizer state all parameters",
        },
        "objective": {
            "C": "sum(c*fresh*legal_CE)/sum(c*fresh) + 0.3*sum(fresh*Huber)/512",
            "P": "sum(c*fresh*legal_CE)/sum(c*fresh) + 0*sum(fresh*Huber)/512",
            "huber_delta": 1.0,
            "training_population": "all expanded training occurrences with multiplicity",
        },
        "cohorts": {
            "fresh": "source_ids==0; preserve duplicate occurrences and own targets",
            "pooled": "all authenticated evaluation occurrences",
        },
        "weighting": {
            "exposure_weighted": "mean across occurrences",
            "equal_input": "within exact input identity mean then across identities",
        },
        "gates": {
            "P fresh CE improvement vs initializer >= 0.001 under both weightings": True,
            "P fresh CE improvement vs C >= 0.001 under both weightings": True,
            "P pooled CE regression <= 0.005 under both weightings": True,
            "P pooled MSE regression <= 0.01 under both weightings": True,
            "P training policy CE decreases": True,
            "exact C reproduction": True,
        },
        "decision_rules": {
            "pass": "advance_to_fresh_confirmation",
            "fail": "close_reset_adam_policy_only_branch",
            "training_J": "descriptive only; policy CE is objective-decrease gate",
            "authorization": "separately designed fresh confirmation only",
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "registration.json").write_text(
        json.dumps(reg, indent=2, sort_keys=True) + "\n"
    )
    return reg


def run_arm(root: Path, arm: str) -> None:
    root = root.resolve()
    out = root / OUT
    reg = json.loads((out / "registration.json").read_text())
    if arm not in ("C", "P") or reg.get("status") != "frozen_before_execution":
        raise ValueError("invalid_execution_state")
    for rel, expected in reg["source_closure_sha256"].items():
        if sha(root / rel) != expected:
            raise ValueError(f"source_changed:{rel}")
    for key, expected in reg["inputs_sha256"].items():
        if sha(root / INPUTS[key]) != expected:
            raise ValueError(f"input_changed:{key}")
    if sha(root / SEED473_OUT / "R-steps.npz") != reg["seed473_R_steps_sha256"]:
        raise ValueError("seed473_R_reference_changed")
    if arm == "P":
        if (
            not (out / "validity-gate.json").is_file()
            or json.loads((out / "validity-gate.json").read_text()).get("passed")
            is not True
        ):
            raise ValueError("exact_C_reproduction_required_before_P")
    arrays = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr = arrays["training"]
    rows = tr["expanded_replay_indices"][
        tr["train_positions"][
            np.asarray(reg["minibatches"]["positions"], dtype=np.int64)
        ]
    ]
    x, targets, masks = (
        tr["inputs"][rows],
        tr["policy_targets"][rows],
        tr["legal_masks"][rows],
    )
    coeff, fresh = (
        tr["policy_coefficients"][rows],
        tr["fresh_source_mask"][rows].astype(np.float32),
    )
    values = tr["value_targets"][rows].reshape(-1)
    torch.set_num_threads(1)
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, root / INIT)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(
        params, lr=0.0000625, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    archive: dict[str, np.ndarray] = {}
    records = []
    for step in range(16):
        optimizer.state.clear()
        sl = slice(step * 512, (step + 1) * 512)
        batch, target, mask = (
            torch.from_numpy(a[sl].copy()) for a in (x, targets, masks)
        )
        c, f, value_target = (
            torch.from_numpy(a[sl].copy()) for a in (coeff, fresh, values)
        )
        pre = [p.detach().clone() for p in params]
        optimizer.zero_grad(set_to_none=True)
        logits, prediction = model(batch)
        ce = -(target * F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)).sum(
            1
        )
        mass = (c * f).sum()
        policy = (ce * c * f).sum() / mass
        huber = F.smooth_l1_loss(prediction.reshape(-1), value_target, reduction="none")
        value = (huber * f).sum() / 512
        objective = policy + (0.3 if arm == "C" else 0.0) * value
        objective.backward()
        for parameter in params:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        gradients = [p.grad.detach().clone() for p in params]
        optimizer.step()
        records.append(
            {
                "step": step + 1,
                "objective": float(objective.detach()),
                "policy_loss": float(policy.detach()),
                "value_loss": float(value.detach()),
                "value_coefficient": 0.3 if arm == "C" else 0.0,
                "policy_mass": float(mass),
                "preclip_norm": float(norm),
                "clipped": bool(float(norm) > 1),
                "optimizer_step": 1,
                "reset_before_update": True,
                "occurrence_indices": rows[sl].tolist(),
            }
        )
        for index, (parameter, gradient, before) in enumerate(
            zip(params, gradients, pre, strict=True)
        ):
            state = optimizer.state[parameter]
            for label, tensor in (
                ("pre", before),
                ("gradient", gradient),
                ("post", parameter.detach()),
                ("exp_avg", state["exp_avg"]),
                ("exp_avg_sq", state["exp_avg_sq"]),
            ):
                archive[f"{step:02d}_{label}_{index:02d}"] = tensor.cpu().numpy().copy()
            archive[f"{step:02d}_displacement_{index:02d}"] = (
                (parameter.detach().double() - before.double()).cpu().numpy().copy()
            )
    np.savez_compressed(out / f"{arm}-steps.npz", **archive)
    (out / f"{arm}-trajectory.json").write_text(
        json.dumps({"arm": arm, "steps": records}, indent=2, sort_keys=True) + "\n"
    )
    if arm == "C":
        try:
            compare_control(root / SEED473_OUT / "R-steps.npz", out / "C-steps.npz")
            passed, mismatch = True, None
        except (ValueError, KeyError) as exc:
            passed, mismatch = False, str(exc)
        gate = {
            "schema": "seed474-exact-C-gate-v1",
            "passed": passed,
            "first_mismatch": mismatch,
            "seed473_R_steps_sha256": sha(root / SEED473_OUT / "R-steps.npz"),
            "C_steps_sha256": sha(out / "C-steps.npz"),
            "compared": "all archived state tensors and all sixteen post-step parameter states",
        }
        (out / "validity-gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n"
        )
        if not passed:
            raise ValueError(mismatch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run-C", "run-P"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root)
    else:
        run_arm(args.root, args.action[-1])


if __name__ == "__main__":
    main()
