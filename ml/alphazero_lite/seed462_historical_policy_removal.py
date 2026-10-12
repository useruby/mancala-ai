"""Prospective historical-policy-removal ablation (seed462)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed435_adam_direction as seed435
from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed462-historical-policy-removal"
INIT_SHA = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
CONTROL = "docs/data/seed442-kl-capped-adam-screen"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(root: Path) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    seed435.ROOT = root
    seed435.INIT = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    arrays = seed435._inputs()
    provenance = arrays[7]
    fresh = np.asarray([row["source"] == "fresh" for row in provenance], dtype=bool)
    return arrays, fresh


def census(
    arrays: tuple[np.ndarray, ...], fresh: np.ndarray, positions: list[int]
) -> list[dict[str, Any]]:
    replay = arrays[3][arrays[5]]
    coeff = arrays[4]
    rows = []
    for step in range(16):
        expanded = np.asarray(positions[step * 512 : (step + 1) * 512], dtype=np.int64)
        ids = replay[expanded]
        mass = float(coeff[ids][fresh[ids]].sum(dtype=np.float64))
        count = int(fresh[ids].sum())
        if mass <= 0:
            raise ValueError(f"empty_fresh_policy_denominator:{step + 1}")
        rows.append(
            {
                "step": step + 1,
                "positions": expanded.tolist(),
                "compact_rows": ids.tolist(),
                "fresh_occurrences": count,
                "fresh_positive_coefficient_mass": mass,
            }
        )
    return rows


def policy_objective(
    ce: torch.Tensor, coefficient: torch.Tensor, fresh: torch.Tensor, arm: str
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = coefficient if arm == "C" else coefficient * fresh.to(coefficient.dtype)
    denominator = weights.sum()
    if float(denominator) <= 0:
        raise ValueError("empty_policy_denominator")
    return (weights * ce).sum() / denominator, denominator


def register(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT.relative_to(ROOT)
    if (out / "registration.json").exists():
        raise ValueError("registration_is_immutable")
    seed435.ROOT = root
    arrays, fresh = inputs(root)
    reg_path = root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    reg = json.loads(reg_path.read_text())
    freeze = root / "docs/data/seed416-policy-target-softening/training-freeze-v3"
    with gzip.open(freeze / "epoch-permutations.json.gz", "rt") as f:
        order = [int(x) for x in json.load(f)[0][:8192]]
    control = json.loads((root / CONTROL / "registration.json").read_text())
    if order != control["first_16_minibatch_positions"]:
        raise ValueError("seed442_minibatch_vector_mismatch")
    rows = census(arrays, fresh, order)
    if len(rows) != 16 or len(order) != 8192:
        raise ValueError("invalid_census")
    paths = [
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz",
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        *[
            f"docs/data/seed426-canonical-overlap/sources/{item['name']}.jsonl.gz"
            for item in reg["replays"]
        ],
        *[
            f"{CONTROL}/{name}"
            for name in (
                "registration.json",
                "receipt.json",
                "evidence.json",
                "step-tensors.npz",
                "A-final.npz",
            )
        ],
    ]
    sources = [
        "ml/alphazero_lite/seed462_historical_policy_removal.py",
        "ml/alphazero_lite/verify_seed462_historical_policy_removal.py",
        "ml/alphazero_lite/test_seed462_historical_policy_removal.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/seed435_adam_direction.py",
        "ml/alphazero_lite/seed429_policy_normalization.py",
        "ml/alphazero_lite/verify_seed434_census_source.py",
    ]
    snapshots = out / "source-snapshots"
    for relative in sources:
        snapshot = snapshots / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            raise ValueError(f"source_snapshot_exists:{relative}")
        shutil.copyfile(root / relative, snapshot)
    result = {
        "schema": "seed462-registration-v1",
        "status": "frozen_before_C",
        "hypothesis": "historical policy targets contribute to fresh unseen policy regression while value supervision is retained",
        "initializer_sha256": INIT_SHA,
        "source_sha256": {p: sha(root / p) for p in sources},
        "input_sha256": {p: sha(root / p) for p in paths},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "configuration": {
            "architecture": "residual_v3",
            "hidden_sizes": [96, 3],
            "input_size": 27,
            "batch_size": 512,
            "steps": 16,
            "adam": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0,
            },
            "clip_norm": 1.0,
            "value_coefficient": 0.3,
            "value": "mean Huber delta=1 over every occurrence",
            "C_policy": "sum(c*legal_CE)/sum(c)",
            "F_policy": "sum(c*fresh*legal_CE)/sum(c*fresh)",
            "projection": None,
            "backtracking": None,
        },
        "first_16_minibatch_positions": order,
        "census": rows,
        "evaluation_membership_sha256": sha(root / paths[3]),
        "decision_rules": {
            "fresh_ce_improvement_vs_initializer_and_C": 0.005,
            "pooled_ce_max_regression_vs_initializer": 0.002,
            "pooled_and_fresh_mse_max_regression_vs_initializer": 0.002,
            "full_training_fresh_policy_objective_decreases": True,
            "exact_C_reproduction_required": True,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "census.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (out / "registration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def execute(root: Path, arm: str) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT.relative_to(ROOT)
    protocol = json.loads((out / "registration.json").read_text())
    if protocol["status"] != "frozen_before_C" or arm not in ("C", "F"):
        raise ValueError("invalid_execution_state")
    if arm == "F" and not (out / "C-verified.json").exists():
        raise ValueError("F_requires_verified_C")
    for path, digest in protocol["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"source_changed:{path}")
    for path, digest in protocol["input_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"input_changed:{path}")
    arrays, fresh = inputs(root)
    x, policy, value, replay, coeff, train_positions = arrays[:6]
    train_replay = replay[train_positions]
    positions = protocol["first_16_minibatch_positions"]
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        model,
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    trace: list[dict[str, Any]] = []
    tensors: dict[str, np.ndarray] = {}
    for step in range(16):
        expanded = np.asarray(positions[step * 512 : (step + 1) * 512], dtype=np.int64)
        ids = train_replay[expanded]
        before = [p.detach().clone() for p in params]
        optimizer.zero_grad(set_to_none=True)
        xb = torch.from_numpy(x[ids])
        pb = torch.from_numpy(policy[ids])
        vb = torch.from_numpy(value[ids])
        cb = torch.from_numpy(coeff[ids])
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ids]))
        logits, pred = model(xb)
        ce = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), pb
        )
        mask = torch.from_numpy(fresh[ids]).to(cb.dtype)
        try:
            policy_loss, denom = policy_objective(ce, cb, mask, arm)
        except ValueError as error:
            raise ValueError(f"empty_policy_denominator:{step + 1}") from error
        value_loss = train.compute_value_loss_vector(
            pred, vb, value_loss="huber", huber_delta=1.0
        ).mean()
        loss = policy_loss + 0.3 * value_loss
        loss.backward()
        preclip = torch.nn.utils.clip_grad_norm_(params, 1.0)
        grads = [p.grad.detach().clone() for p in params]
        optimizer.step()
        proposal = [p.detach().clone() for p in params]
        post = proposal
        trace.append(
            {
                "step": step + 1,
                "positions": expanded.tolist(),
                "rows": ids.tolist(),
                "policy_mass": float(denom),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "total_loss": float(loss.detach()),
                "preclip_norm": float(preclip),
                "clipped": bool(float(preclip) > 1.0),
                "grad_norm": float(
                    torch.sqrt(sum(g.double().square().sum() for g in grads))
                ),
                "pre_hash": _tensor_hash(before),
                "post_hash": _tensor_hash(post),
            }
        )
        for j, (p0, g, p1) in enumerate(zip(before, grads, post, strict=True)):
            tensors[f"{step:02d}_pre_{j:02d}"] = p0.cpu().numpy().copy()
            tensors[f"{step:02d}_gradient_{j:02d}"] = g.cpu().numpy().copy()
            tensors[f"{step:02d}_proposal_{j:02d}"] = p1.cpu().numpy().copy()
            tensors[f"{step:02d}_post_{j:02d}"] = p1.cpu().numpy().copy()
            st = optimizer.state[params[j]]
            tensors[f"{step:02d}_exp_avg_{j:02d}"] = st["exp_avg"].cpu().numpy().copy()
            tensors[f"{step:02d}_exp_avg_sq_{j:02d}"] = (
                st["exp_avg_sq"].cpu().numpy().copy()
            )
    np.savez_compressed(out / f"{arm}-steps.npz", **tensors)
    result = {
        "arm": arm,
        "steps": trace,
        "final_sha256": _tensor_hash([p.detach() for p in params]),
    }
    (out / f"{arm}-trajectory.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if arm == "C":
        matches = compare_seed442_control(root, out)
        (out / "C-verified.json").write_text(
            json.dumps(matches, indent=2, sort_keys=True) + "\n"
        )
    return result


def compare_seed442_control(root: Path, out: Path) -> dict[str, Any]:
    """Exact selected-update and post-parameter comparison against seed442 A."""
    mismatches: list[str] = []
    with (
        np.load(out / "C-steps.npz", allow_pickle=False) as actual,
        np.load(root / CONTROL / "step-tensors.npz", allow_pickle=False) as expected,
    ):
        for step in range(16):
            for index in range(
                sum(key.startswith(f"{step:02d}_post_") for key in actual.files)
            ):
                ours = actual[f"{step:02d}_post_{index:02d}"]
                theirs = expected[f"A_{step:02d}_post_{index:02d}"]
                if not np.array_equal(ours, theirs):
                    mismatches.append(f"post:{step + 1}:{index}")
                proposal = actual[f"{step:02d}_proposal_{index:02d}"]
                control_proposal = expected[f"A_{step:02d}_proposal_{index:02d}"]
                if not np.array_equal(proposal, control_proposal):
                    mismatches.append(f"update:{step + 1}:{index}")
    if mismatches:
        raise ValueError(f"seed442_C_reproduction_mismatch:{mismatches[:8]}")
    return {
        "status": "exact",
        "steps": 16,
        "parameter_tensors_per_step": 22,
        "mismatches": [],
    }


def _tensor_hash(values: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "run-C", "run-F"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "register":
        register(args.root)
    else:
        execute(args.root, args.action[-1])


if __name__ == "__main__":
    main()
