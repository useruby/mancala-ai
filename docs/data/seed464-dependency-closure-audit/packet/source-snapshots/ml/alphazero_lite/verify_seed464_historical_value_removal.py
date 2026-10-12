"""Independent, read-only semantic verifier for seed464."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed464_historical_value_removal as experiment
from ml.alphazero_lite import train


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / experiment.OUT
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    if registration.get("status") != "frozen_before_C":
        raise ValueError("registration_not_frozen")
    for path, expected in registration["source_sha256"].items():
        if digest(root / path) != expected:
            raise ValueError(f"source_hash_mismatch:{path}")
    for path, expected in registration["input_sha256"].items():
        if digest(root / path) != expected:
            raise ValueError(f"input_hash_mismatch:{path}")
    arrays, fresh = experiment.datasets(root)
    x, policies, values, replay, coeff, train_positions = arrays[:6]
    expanded = replay[train_positions]
    expected_positions = registration["minibatches"]["positions"]
    reconstructed: dict[str, Any] = {}
    for arm in ("C", "T"):
        trajectory = json.loads((out / f"{arm}-trajectory.json").read_text())
        if len(trajectory["steps"]) != 16:
            raise ValueError(f"step_count:{arm}")
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(model, root / experiment.INIT)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
        )
        with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as archive:
            for step in range(16):
                record = trajectory["steps"][step]
                positions = np.asarray(
                    expected_positions[step * 512 : (step + 1) * 512], dtype=np.int64
                )
                rows = expanded[positions]
                if (
                    record["positions"] != positions.tolist()
                    or record["rows"] != rows.tolist()
                ):
                    raise ValueError(f"minibatch_mismatch:{arm}:{step + 1}")
                optimizer.zero_grad(set_to_none=True)
                ids = rows
                legal = torch.from_numpy(
                    train.legal_mask_matrix_for_encoded_states(x[ids])
                )
                logits, pred = model(torch.from_numpy(x[ids]))
                ce = train.compute_policy_cross_entropy(
                    logits.masked_fill(legal <= 0, -1e9),
                    torch.from_numpy(policies[ids]),
                )
                c = torch.from_numpy(coeff[ids])
                mask = torch.from_numpy(fresh[ids])
                objective_policy, mass = experiment.policy_loss(ce, c, mask)
                huber = train.compute_value_loss_vector(
                    pred,
                    torch.from_numpy(values[ids]),
                    value_loss="huber",
                    huber_delta=1.0,
                )
                value_term = (
                    huber if arm == "C" else huber * mask.to(huber.dtype)
                ).sum() / 512.0
                total = objective_policy + 0.3 * value_term
                if (
                    float(mass) != record["policy_mass"]
                    or float(objective_policy) != record["policy_loss"]
                    or float(value_term) != record["value_loss"]
                ):
                    raise ValueError(f"objective_mismatch:{arm}:{step + 1}")
                before = [
                    parameter.detach().cpu().numpy().copy() for parameter in params
                ]
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not np.isclose(
                    float(norm), record["preclip_norm"], rtol=0, atol=1e-7
                ):
                    raise ValueError(f"clip_mismatch:{arm}:{step + 1}")
                grads = [p.grad.detach().clone() for p in params]
                optimizer.step()
                for i, (p, grad) in enumerate(zip(params, grads, strict=True)):
                    expected_map = {
                        "pre": archive[f"{step:02d}_pre_{i:02d}"],
                        "gradient": archive[f"{step:02d}_gradient_{i:02d}"],
                        "proposal": archive[f"{step:02d}_proposal_{i:02d}"],
                        "post": archive[f"{step:02d}_post_{i:02d}"],
                        "exp_avg": archive[f"{step:02d}_exp_avg_{i:02d}"],
                        "exp_avg_sq": archive[f"{step:02d}_exp_avg_sq_{i:02d}"],
                    }
                    actual_map = {
                        "pre": before[i],
                        "gradient": grad.detach().cpu().numpy(),
                        "proposal": p.detach().cpu().numpy(),
                        "post": p.detach().cpu().numpy(),
                        "exp_avg": optimizer.state[p]["exp_avg"].detach().cpu().numpy(),
                        "exp_avg_sq": optimizer.state[p]["exp_avg_sq"]
                        .detach()
                        .cpu()
                        .numpy(),
                    }
                    for key, expected in expected_map.items():
                        if not np.array_equal(actual_map[key], expected):
                            raise ValueError(
                                f"tensor_mismatch:{arm}:{step + 1}:{key}:{i}"
                            )
        if trajectory["final_sha256"] != experiment._hash([p.detach() for p in params]):
            raise ValueError(f"final_parameter_hash_mismatch:{arm}")
        reconstructed[arm] = {
            "steps": 16,
            "trajectory_sha256": digest(out / f"{arm}-trajectory.json"),
            "tensor_archive_sha256": digest(out / f"{arm}-steps.npz"),
        }
    # Seed462 F is the pre-registered full-value, fresh-policy control trajectory.
    with (
        np.load(root / experiment.OLD / "F-steps.npz", allow_pickle=False) as control,
        np.load(out / "C-steps.npz", allow_pickle=False) as candidate,
    ):
        for step in range(16):
            for i in range(
                sum(k.startswith(f"{step:02d}_post_") for k in candidate.files)
            ):
                for kind in ("proposal", "post"):
                    if not np.array_equal(
                        candidate[f"{step:02d}_{kind}_{i:02d}"],
                        control[f"{step:02d}_{kind}_{i:02d}"],
                    ):
                        raise ValueError(
                            f"seed462_F_control_mismatch:{step + 1}:{kind}:{i}"
                        )
    # Preservation invariant covers every published experiment file byte-for-byte and by mtime.
    receipt_path = out / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for relative, expected in receipt["publication_sha256"].items():
        if digest(out / relative) != expected:
            raise ValueError(f"publication_hash_mismatch:{relative}")
    result = {
        "status": "valid",
        "control": "exact_seed462_F",
        "arms": reconstructed,
        "publication": "receipt_bound",
        "gates": receipt["decision"]["gates"],
        "decision": receipt["decision"]["status"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
