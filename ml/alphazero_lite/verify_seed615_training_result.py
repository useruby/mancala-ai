"""Independent structural verification of a completed Seed 615 training run."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed615_training_executor as contract
from ml.alphazero_lite.verify_seed615_training_freeze import (
    verify as verify_execution_freeze,
)

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TENSORS = {
    "w_input": ((27, 96), np.dtype("float32")),
    "b_input": ((96,), np.dtype("float32")),
    **{
        f"w_residual_{block}_{layer}": ((96, 96), np.dtype("float32"))
        for block in range(1, 4)
        for layer in range(1, 3)
    },
    **{
        f"b_residual_{block}_{layer}": ((96,), np.dtype("float32"))
        for block in range(1, 4)
        for layer in range(1, 3)
    },
    "w_policy_hidden": ((96, 96), np.dtype("float32")),
    "b_policy_hidden": ((96,), np.dtype("float32")),
    "w_policy": ((96, 6), np.dtype("float32")),
    "b_policy": ((6,), np.dtype("float32")),
    "w_value_hidden": ((96, 48), np.dtype("float32")),
    "b_value_hidden": ((48,), np.dtype("float32")),
    "w_value": ((48, 1), np.dtype("float32")),
    "b_value": ((1,), np.dtype("float32")),
}
FROZEN_TENSORS = {"w_value_hidden", "b_value_hidden", "w_value", "b_value"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_digest(expected: str, path: Path, label: str) -> None:
    if expected != digest(path):
        raise ValueError(f"training_artifact_hash_mismatch:{label}")


def verify_result(output: Path, *, root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    if output != (root / contract.OUTPUT).resolve():
        raise ValueError("noncanonical_output_directory")
    if (
        verify_execution_freeze(root, allow_execution_artifacts=True).get("passed")
        is not True
    ):
        raise ValueError("execution_freeze_not_independently_verified")
    freeze_raw = (root / contract.FREEZE).read_bytes()
    authorization_path = root / contract.AUTHORIZATION
    authorization_raw = authorization_path.read_bytes()
    authorization = json.loads(authorization_raw)
    owner = authorization.get("authorization", {})
    if (
        authorization.get("schema") != "seed615-training-owner-authorization-v1"
        or authorization.get("status") != "authorized-for-training"
        or authorization.get("execution_freeze_sha256")
        != hashlib.sha256(freeze_raw).hexdigest()
        or owner.get("type") != "explicit-project-owner-authorization"
        or not isinstance(owner.get("authorized_by"), str)
        or not owner["authorized_by"].strip()
        or owner.get("training_may_start") is not True
    ):
        raise ValueError("training_authorization_or_freeze_binding_invalid")
    if not output.is_dir():
        raise ValueError("training_output_missing")
    expected_artifacts = {
        "training-trace.jsonl",
        "seed615-S615-update-512.npz",
        "training-receipt.json",
        "training-integrity-report.json",
    }
    if {item.name for item in output.iterdir()} != expected_artifacts:
        raise ValueError("training_output_artifact_set_mismatch")
    trace_path = output / "training-trace.jsonl"
    checkpoint_path = output / "seed615-S615-update-512.npz"
    receipt_path = output / "training-receipt.json"
    report_path = output / "training-integrity-report.json"
    receipt = json.loads(receipt_path.read_bytes())
    report = json.loads(report_path.read_bytes())
    schedule = json.loads(
        (
            root / Path("docs/data/seed599-replay-coverage/batch-schedule-B.json")
        ).read_bytes()
    )
    if (
        len(schedule) != 512
        or hashlib.sha256(contract.canonical_bytes(schedule)).hexdigest()
        != contract.SCHEDULE_SHA256
    ):
        raise ValueError("registered_schedule_length_mismatch")
    if (
        receipt.get("schema") != "seed615-training-result-v1"
        or receipt.get("execution_freeze_sha256")
        != hashlib.sha256(freeze_raw).hexdigest()
        or receipt.get("authorization_sha256") != digest(root / contract.AUTHORIZATION)
        or receipt.get("registration_sha256") != digest(root / contract.REGISTRATION)
        or receipt.get("parent_checkpoint_sha256") != contract.PARENT_SHA256
        or receipt.get("b512_checkpoint_sha256") != contract.B512_SHA256
        or receipt.get("membership_sha256") != contract.MEMBERSHIP_SHA256
        or receipt.get("batch_schedule_sha256") != contract.SCHEDULE_SHA256
        or receipt.get("updates") != 512
        or receipt.get("runtime") != contract.RUNTIME
    ):
        raise ValueError("training_receipt_binding_mismatch")
    if (
        authorization.get("execution_freeze_sha256")
        != hashlib.sha256(freeze_raw).hexdigest()
    ):
        raise ValueError("authorization_freeze_binding_mismatch")
    if (
        receipt.get("checkpoint") != checkpoint_path.name
        or receipt.get("trace") != trace_path.name
    ):
        raise ValueError("training_artifact_name_mismatch")
    for field, path in (
        ("checkpoint_sha256", checkpoint_path),
        ("trace_sha256", trace_path),
        ("integrity_report_sha256", report_path),
    ):
        require_digest(receipt.get(field), path, field)
    lines = trace_path.read_bytes().splitlines()
    if len(lines) != 512:
        raise ValueError("training_trace_length_mismatch")
    for update, (line, expected_indexes) in enumerate(
        zip(lines, schedule, strict=True), 1
    ):
        record = json.loads(line)
        if record.get("update") != update or record.get("indexes") != expected_indexes:
            raise ValueError(f"training_trace_schedule_mismatch:{update}")
        if record.get("seed") != 599 or record.get("runtime") != contract.RUNTIME:
            raise ValueError(f"training_trace_runtime_or_seed_mismatch:{update}")
        if not math.isfinite(record.get("loss", math.nan)) or not math.isfinite(
            record.get("gradient_norm", math.nan)
        ):
            raise ValueError(f"training_trace_nonfinite:{update}")
    trainable_names = sorted(
        [
            "input_layer.weight",
            "input_layer.bias",
            "policy_hidden_layer.weight",
            "policy_hidden_layer.bias",
            "policy_head.weight",
            "policy_head.bias",
        ]
        + [
            f"residual_layers.{block}.{layer}.{field}"
            for block in range(3)
            for layer in range(2)
            for field in ("weight", "bias")
        ]
    )
    with (
        np.load(checkpoint_path, allow_pickle=False) as checkpoint,
        np.load(root / contract.PARENT, allow_pickle=False) as parent,
        np.load(root / contract.B512, allow_pickle=False) as b512,
    ):
        if (
            set(checkpoint.files) != set(EXPECTED_TENSORS)
            or len(checkpoint.files) != 22
        ):
            raise ValueError("checkpoint_tensor_names_mismatch")
        for name, (shape, dtype) in EXPECTED_TENSORS.items():
            actual = checkpoint[name]
            if (
                actual.shape != shape
                or actual.dtype != dtype
                or not np.isfinite(actual).all()
            ):
                raise ValueError(f"checkpoint_tensor_schema_or_finite_mismatch:{name}")
        if any(
            not np.array_equal(checkpoint[key], parent[key]) for key in FROZEN_TENSORS
        ):
            raise ValueError("frozen_checkpoint_tensor_changed")
        equality = {
            name: bool(np.array_equal(checkpoint[name], b512[name]))
            for name in sorted(EXPECTED_TENSORS)
        }
        movement = {
            name: bool(
                not np.array_equal(
                    checkpoint[contract._tensor_key(name)],
                    parent[contract._tensor_key(name)],
                )
            )
            for name in trainable_names
        }
    adam = report.get("adam_state", {})
    steps = adam.get("step_counters")
    if (
        report.get("schema") != "seed615-training-integrity-report-v1"
        or report.get("updates") != 512
        or not isinstance(steps, list)
        or adam.get("parameter_count") != len(trainable_names)
        or [item.get("parameter") for item in steps] != trainable_names
        or any(item.get("step") != 512 for item in steps)
        or not re.fullmatch(r"[0-9a-f]{64}", adam.get("state_sha256", ""))
        or report.get("frozen_parameters_unchanged") is not True
        or report.get("b512_tensor_equality_diagnostic") != equality
        or report.get("b512_equal_tensor_names")
        != [name for name, equal in equality.items() if equal]
        or report.get("b512_exact_tensor_equality") != all(equality.values())
        or report.get("b512_equality_used_for_selection") is not False
        or report.get("trainable_parameter_movement_diagnostic") != movement
        or report.get("all_trainable_parameters_moved") != all(movement.values())
    ):
        raise ValueError("training_integrity_report_invalid")
    if not isinstance(report.get("trainable_parameter_movement_diagnostic"), dict):
        raise ValueError("trainable_movement_diagnostic_missing")
    return {
        "passed": True,
        "updates": 512,
        "checkpoint_sha256": digest(checkpoint_path),
        "trace_sha256": digest(trace_path),
        "checkpoint_tensor_count": len(EXPECTED_TENSORS),
        "training_authorized": authorization.get("status") == "authorized-for-training",
        "numerical_reproduction_claim": False,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_result(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
