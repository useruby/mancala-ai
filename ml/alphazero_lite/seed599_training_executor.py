"""Guarded executor for the frozen Seed 599 training arms.

Importing this module does not train or create artifacts.  Training is available
only when a separately supplied owner authorization binds the execution freeze.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import self_play, train
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed599_replay_coverage import (
    ROOT,
    canonical_bytes,
    read_only_preflight,
)

DATA = Path("docs/data/seed599-replay-coverage")
FREEZE = DATA / "training-execution-freeze.json"
AUTHORIZATION = DATA / "training-authorization.json"
PARENT = Path("docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/model.npz")
PARENT_SHA256 = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
TRAINABLE = ("input_layer.", "residual_layers.", "policy_hidden_layer.", "policy_head.")
FROZEN = ("value_hidden_layer.", "value_head.")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 1,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "deterministic_algorithms": True,
    }


def load_freeze(root: Path = ROOT) -> tuple[dict[str, Any], bytes]:
    path = root / FREEZE
    raw = path.read_bytes()
    freeze = json.loads(raw)
    if freeze.get("schema") != "seed599-training-execution-freeze-v1":
        raise ValueError("execution_freeze_schema_mismatch")
    if freeze.get("status") != "frozen-training-not-authorized":
        raise ValueError("execution_freeze_status_mismatch")
    if root == ROOT:
        for relative, expected in freeze.get("source_sha256", {}).items():
            if sha256(root / relative) != expected:
                raise ValueError(f"execution_source_hash_mismatch:{relative}")
        if set(freeze.get("source_sha256", {})) != {
            "ml/alphazero_lite/seed599_training_executor.py",
            "ml/alphazero_lite/verify_seed599_training_result.py",
            "ml/alphazero_lite/seed599_replay_coverage.py",
        }:
            raise ValueError("execution_source_hash_set_mismatch")
    return freeze, raw


def verify_authorization(root: Path = ROOT) -> dict[str, Any]:
    freeze, raw = load_freeze(root)
    authorization_path = root / AUTHORIZATION
    authorization = json.loads(authorization_path.read_bytes())
    project_owner = authorization.get("authorization", {})
    if (
        authorization.get("schema") != "seed599-training-owner-authorization-v1"
        or authorization.get("status") != "authorized-for-training"
        or project_owner.get("type") != "explicit-project-owner-authorization"
        or project_owner.get("training_may_start") is not True
        or not isinstance(project_owner.get("authorized_by"), str)
        or not project_owner["authorized_by"].strip()
    ):
        raise ValueError("training_not_authorized")
    expected = hashlib.sha256(raw).hexdigest()
    if authorization.get("execution_freeze_sha256") != expected:
        raise ValueError("authorization_freeze_binding_mismatch")
    return authorization


def _load_registered_arm(
    arm: str, root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    if arm not in ("A", "B"):
        raise ValueError("invalid_training_arm")
    data = root / DATA
    config = json.loads((data / f"training-{arm}.json").read_bytes())
    freeze, _raw = load_freeze(root)
    if (
        sha256(data / f"training-{arm}.json")
        != freeze["arms"][arm]["configuration_sha256"]
    ):
        raise ValueError("training_configuration_mismatch")
    if config.get("status") != "frozen-pretraining-not-authorized":
        raise ValueError("training_configuration_status_mismatch")
    membership_name = (
        "control-membership.json" if arm == "A" else "treatment-membership.json"
    )
    rows = json.loads((data / membership_name).read_bytes())
    if (
        len(rows) != config["states"]
        or hashlib.sha256(canonical_bytes(rows)).hexdigest()
        != config["membership_sha256"]
    ):
        raise ValueError("training_membership_identity_mismatch")
    batches = json.loads((data / f"batch-schedule-{arm}.json").read_bytes())
    if (
        len(batches) != 512
        or hashlib.sha256(canonical_bytes(batches)).hexdigest()
        != config["batch_schedule_sha256"]
    ):
        raise ValueError("training_batch_schedule_identity_mismatch")
    for batch in batches:
        if (
            len(batch) != 32
            or len(set(batch)) != 32
            or any(
                type(index) is not int or not 0 <= index < len(rows) for index in batch
            )
        ):
            raise ValueError("invalid_batch_schedule")
    return config, rows, batches


def _check_runtime(expected: dict[str, Any]) -> None:
    actual = runtime_identity()
    if actual != expected:
        raise ValueError(f"runtime_mismatch:{actual}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def train_arm(arm: str, output: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Run exactly one registered arm; refuse before mkdir unless authorized."""
    if output.exists():
        raise ValueError("output_directory_exists")
    freeze, _raw = load_freeze(root)
    verify_authorization(root)
    _check_runtime(freeze["runtime"])
    read_only_preflight(root)
    config, rows, batches = _load_registered_arm(arm, root)
    if sha256(root / PARENT) != PARENT_SHA256:
        raise ValueError("parent_checkpoint_hash_mismatch")
    train.set_seed(599)
    model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    if train.load_checkpoint_into_model(model, root / PARENT):
        raise ValueError("parent_checkpoint_tensor_mismatch")
    frozen_before: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        trainable = name.startswith(TRAINABLE)
        if not trainable and not name.startswith(FROZEN):
            raise ValueError(f"unregistered_parameter_group:{name}")
        parameter.requires_grad_(trainable)
        if not trainable:
            frozen_before[name] = parameter.detach().clone()
    x = torch.tensor(
        [
            self_play.encode_state(row["canonical_state"], input_encoding="kalah_v3")
            for row in rows
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([row["target"] for row in rows], dtype=torch.float32)
    legal = torch.zeros((len(rows), 6), dtype=torch.bool)
    for index, row in enumerate(rows):
        moves = KalahGame.from_state(row["canonical_state"]).possible_moves()
        if moves != row["legal_moves"] or any(
            row["target"][action] != 0 for action in range(6) if action not in moves
        ):
            raise ValueError("legal_mask_or_target_mismatch")
        legal[index, moves] = True
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    output.mkdir(parents=True)
    trace_path = output / "training-trace.jsonl"
    with trace_path.open("x", encoding="utf-8") as trace:
        for update, indexes in enumerate(batches, 1):
            index = torch.tensor(indexes, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x[index])
            masked = logits.masked_fill(~legal[index], torch.finfo(logits.dtype).min)
            loss = (
                -(targets[index] * torch.log_softmax(masked, dim=1)).sum(dim=1).mean()
            )
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite_loss:{update}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if any(
                not torch.isfinite(parameter).all() for parameter in model.parameters()
            ):
                raise ValueError(f"nonfinite_parameter:{update}")
            trace.write(
                json.dumps(
                    {
                        "update": update,
                        "indexes": indexes,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(norm),
                        "seed": 599,
                        "runtime": freeze["runtime"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    parameters = dict(model.named_parameters())
    if any(
        not torch.equal(parameters[name].detach(), before)
        for name, before in frozen_before.items()
    ):
        raise ValueError("frozen_parameter_changed")
    checkpoint = output / f"seed599-arm-{arm}-update-512.npz"
    np.savez(checkpoint, **train.checkpoint_from_model(model))
    receipt = {
        "schema": "seed599-training-result-v1",
        "arm": arm,
        "states": len(rows),
        "updates": len(batches),
        "execution_freeze_sha256": sha256(root / FREEZE),
        "membership_sha256": config["membership_sha256"],
        "batch_schedule_sha256": config["batch_schedule_sha256"],
        "parent_checkpoint_sha256": PARENT_SHA256,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "trace": trace_path.name,
        "trace_sha256": sha256(trace_path),
        "frozen_parameter_names": sorted(frozen_before),
        "frozen_parameters_unchanged": True,
    }
    (output / "training-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("A", "B"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(train_arm(args.arm, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
