"""Seed473 carried-Adam control and per-update moment-reset ablation."""

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
from ml.alphazero_lite.seed471_fresh_only_adam_ablation import (
    INIT,
    INPUTS,
    OUT as SEED471_OUT,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed473-adam-moment-reset")
SOURCES = (
    "ml/alphazero_lite/seed473_adam_moment_reset.py",
    "ml/alphazero_lite/publish_seed473_adam_moment_reset.py",
    "ml/alphazero_lite/verify_seed473_adam_moment_reset.py",
    "ml/alphazero_lite/test_seed473_adam_moment_reset.py",
    "ml/alphazero_lite/seed471_fresh_only_adam_ablation.py",
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


def compare_control(seed471_steps: Path, candidate_steps: Path) -> None:
    """Require bitwise equality of every post-step C parameter with seed471 T."""
    with (
        np.load(seed471_steps, allow_pickle=False) as expected,
        np.load(candidate_steps, allow_pickle=False) as actual,
    ):
        count = sum(key.startswith("00_post_") for key in expected.files)
        for step in range(16):
            for i in range(count):
                key = f"{step:02d}_post_{i:02d}"
                if (
                    key not in expected
                    or key not in actual
                    or not np.array_equal(expected[key], actual[key])
                ):
                    raise ValueError(f"control_reproduction_mismatch:{step + 1}:{i}")


def require_control_gate(gate_path: Path) -> None:
    """Authorize reset-arm execution only after a recorded exact pass."""
    if not gate_path.is_file():
        raise ValueError("C_validity_gate_required_before_R")
    if json.loads(gate_path.read_text()).get("passed") is not True:
        raise ValueError("C_reproduction_gate_failed")


def freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    if out.exists() and any(out.iterdir()):
        raise ValueError("seed473_output_exists")
    arrays = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = arrays["training"], arrays["evaluation"]
    positions = tr["first16_minibatch_positions"][:8192]
    if (
        len(positions) != 8192
        or len(ev["source_ids"]) != 2607
        or int((ev["source_ids"] == 0).sum()) != 859
    ):
        raise ValueError("authenticated_cohort_or_schedule_mismatch")
    reg = {
        "schema": "seed473-adam-moment-reset-registration-v1",
        "status": "frozen_before_execution",
        "interpretation": "exploratory; reused evaluation cohort is not independent confirmation",
        "hypothesis": "accumulated Adam state contributes to seed471 T unfavorable trajectory directions",
        "inputs_sha256": {k: sha(root / v) for k, v in INPUTS.items()},
        "source_closure_sha256": {p: sha(root / p) for p in SOURCES},
        "authenticated_manifest_sha256": MANIFEST_SHA256,
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "initializer_sha256": sha(root / INIT),
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
            "C": {"lr": 0.0000625, "state": "ordinary sequential Adam"},
            "R": {
                "lr": 0.0000625,
                "state": "zero moments and step counter immediately before every update",
            },
        },
        "optimizer": {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "gradient_clip_norm": 1.0,
            "numeric_semantics": "float32 torch; one thread; no norm matching",
        },
        "objective": {
            "J": "sum(c*fresh*legal_CE)/sum(c*fresh) + 0.3*sum(fresh*Huber)/512",
            "huber_delta": 1.0,
            "training_J_population": "all expanded occurrences at frozen train_positions, with multiplicity",
        },
        "cohorts": {
            "fresh": "source_ids==0; occurrence-specific targets and duplicates retained",
            "pooled": "all authenticated evaluation occurrences",
            "fresh_occurrences": 859,
            "pooled_occurrences": 2607,
        },
        "weighting": {
            "exposure_weighted": "arithmetic mean over occurrences",
            "equal_input": "within exact input_identity mean, then across identities",
        },
        "decision_rules": {
            "fresh_ce_improvement_vs_initializer": 0.001,
            "fresh_ce_improvement_vs_C": 0.001,
            "pooled_ce_max_regression_vs_initializer": 0.005,
            "pooled_mse_max_regression_vs_initializer": 0.01,
            "training_J": "R endpoint objective strictly below initializer",
            "decision_pass": "advance_to_fresh_confirmation",
            "decision_fail": "close_moment_reset_branch",
            "control": "all 16 C post-step parameter states exactly equal seed471 T",
            "authorization": "separately designed fresh confirmation only",
        },
        "validity_gate": {
            "status": "pending",
            "control_reproduction": "bitwise equality for all post-step parameter tensors",
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
    if arm not in ("C", "R") or reg.get("status") != "frozen_before_execution":
        raise ValueError("invalid_execution_state")
    for rel, expected in reg["source_closure_sha256"].items():
        if sha(root / rel) != expected:
            raise ValueError(f"source_changed:{rel}")
    for key, expected in reg["inputs_sha256"].items():
        if sha(root / INPUTS[key]) != expected:
            raise ValueError(f"input_changed:{key}")
    if arm == "R":
        if (
            not (out / "C-steps.npz").is_file()
            or not (out / "validity-gate.json").is_file()
        ):
            raise ValueError("C_validity_gate_required_before_R")
        require_control_gate(out / "validity-gate.json")
    arrays = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr = arrays["training"]
    positions = np.asarray(reg["minibatches"]["positions"], dtype=np.int64)
    compact = tr["expanded_replay_indices"][tr["train_positions"][positions]]
    x, policy, masks = (
        tr["inputs"][compact],
        tr["policy_targets"][compact],
        tr["legal_masks"][compact],
    )
    coeff, fresh = (
        tr["policy_coefficients"][compact],
        tr["fresh_source_mask"][compact].astype(np.float32),
    )
    values = tr["value_targets"][compact].reshape(-1)
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, root / INIT)
    params = list(model.parameters())
    opt = torch.optim.Adam(
        params, lr=0.0000625, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    torch.set_num_threads(1)
    archive: dict[str, np.ndarray] = {}
    records = []
    for step in range(16):
        if arm == "R":
            opt.state.clear()
        sl = slice(step * 512, (step + 1) * 512)
        xb, tb, mb = (torch.from_numpy(a[sl].copy()) for a in (x, policy, masks))
        cb, fb, vb = (torch.from_numpy(a[sl].copy()) for a in (coeff, fresh, values))
        pre = [p.detach().clone() for p in params]
        opt.zero_grad(set_to_none=True)
        logits, pred = model(xb)
        ce = -(tb * F.log_softmax(logits.masked_fill(mb == 0, -1e9), dim=1)).sum(1)
        mass = (cb * fb).sum()
        pl = (ce * cb * fb).sum() / mass
        vl = (F.smooth_l1_loss(pred.reshape(-1), vb, reduction="none") * fb).sum() / 512
        obj = pl + 0.3 * vl
        obj.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        grads = [p.grad.detach().clone() for p in params]
        opt.step()
        rec = {
            "step": step + 1,
            "objective": float(obj.detach()),
            "policy_loss": float(pl.detach()),
            "value_loss": float(vl.detach()),
            "policy_mass": float(mass),
            "preclip_norm": float(norm),
            "clipped": bool(float(norm) > 1),
            "optimizer_step": 1 if arm == "R" else step + 1,
            "reset_before_update": arm == "R",
            "occurrence_indices": compact[sl].tolist(),
        }
        records.append(rec)
        for i, (p, g, a) in enumerate(zip(params, grads, pre, strict=True)):
            state = opt.state[p]
            for name, ten in (
                ("pre", a),
                ("gradient", g),
                ("post", p.detach()),
                ("exp_avg", state["exp_avg"]),
                ("exp_avg_sq", state["exp_avg_sq"]),
            ):
                archive[f"{step:02d}_{name}_{i:02d}"] = ten.cpu().numpy().copy()
            archive[f"{step:02d}_displacement_{i:02d}"] = (
                (p.detach().double() - a.double()).cpu().numpy().copy()
            )
    np.savez_compressed(out / f"{arm}-steps.npz", **archive)
    (out / f"{arm}-trajectory.json").write_text(
        json.dumps({"arm": arm, "steps": records}, indent=2, sort_keys=True) + "\n"
    )
    if arm == "C":
        try:
            compare_control(root / SEED471_OUT / "T-steps.npz", out / "C-steps.npz")
            passed, mismatch = True, None
        except ValueError as exc:
            passed, mismatch = False, str(exc)
        gate = {
            "schema": "seed473-control-validity-gate-v1",
            "passed": passed,
            "first_mismatch": mismatch,
            "compared": "16 post-step states x all parameter tensors",
            "seed471_T_steps_sha256": sha(root / SEED471_OUT / "T-steps.npz"),
            "C_steps_sha256": sha(out / "C-steps.npz"),
        }
        (out / "validity-gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n"
        )
        if not passed:
            raise ValueError(mismatch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run-C", "run-R"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root)
    else:
        run_arm(args.root, args.action[-1])


if __name__ == "__main__":
    main()
