"""Independent integrity, Adam-state, and exact-control verifier for seed462."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed462_historical_policy_removal as seed462


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / seed462.OUT.relative_to(seed462.ROOT)
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    if registration.get("status") != "frozen_before_C":
        raise ValueError("registration_not_frozen")
    for name, expected in registration["source_sha256"].items():
        if seed462.sha(root / name) != expected:
            raise ValueError(f"source_hash_mismatch:{name}")
    for name, expected in registration["input_sha256"].items():
        if seed462.sha(root / name) != expected:
            raise ValueError(f"input_hash_mismatch:{name}")
    arrays, fresh = seed462.inputs(root)
    actual_census = seed462.census(
        arrays, fresh, registration["first_16_minibatch_positions"]
    )
    if actual_census != registration["census"]:
        raise ValueError("census_semantic_mismatch")
    control = seed462.compare_seed442_control(root, out)
    result: dict[str, Any] = {
        "status": "valid",
        "census": "valid",
        "C_control": control,
        "arms": {},
    }
    for arm in ("C", "F"):
        trajectory_path = out / f"{arm}-trajectory.json"
        tensor_path = out / f"{arm}-steps.npz"
        if not trajectory_path.exists():
            continue
        trajectory = json.loads(trajectory_path.read_text())
        if len(trajectory["steps"]) != 16:
            raise ValueError(f"step_count:{arm}")
        with np.load(tensor_path, allow_pickle=False) as archive:
            count = sum(key.startswith("00_pre_") for key in archive.files)
            m = [
                torch.zeros_like(torch.from_numpy(archive[f"00_pre_{i:02d}"].copy()))
                for i in range(count)
            ]
            v = [torch.zeros_like(tensor) for tensor in m]
            for step, record in enumerate(trajectory["steps"]):
                grads = [
                    torch.from_numpy(archive[f"{step:02d}_gradient_{i:02d}"].copy())
                    for i in range(count)
                ]
                for i, gradient in enumerate(grads):
                    m[i].lerp_(gradient, 0.1)
                    v[i].mul_(0.999).addcmul_(gradient, gradient, value=0.001)
                    if not np.array_equal(
                        m[i].numpy(), archive[f"{step:02d}_exp_avg_{i:02d}"]
                    ):
                        raise ValueError(f"first_moment_mismatch:{arm}:{step + 1}:{i}")
                    if not np.array_equal(
                        v[i].numpy(), archive[f"{step:02d}_exp_avg_sq_{i:02d}"]
                    ):
                        raise ValueError(f"second_moment_mismatch:{arm}:{step + 1}:{i}")
                if (
                    hashlib.sha256(
                        b"".join(
                            archive[f"{step:02d}_post_{i:02d}"].astype("<f4").tobytes()
                            for i in range(count)
                        )
                    ).hexdigest()
                    != record["post_hash"]
                ):
                    raise ValueError(f"post_hash_mismatch:{arm}:{step + 1}")
        result["arms"][arm] = {
            "steps": 16,
            "optimizer_moments": "reconstructed",
            "trajectory_sha256": seed462.sha(trajectory_path),
            "archive_sha256": seed462.sha(tensor_path),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=seed462.ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
