"""Read-only structural verification of Seed 604 B2048 training results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed604_replay_optimization import (
    DATA,
    ROOT,
    SEED599,
    UPDATES,
    batch_schedule,
    canonical_bytes,
    read_only_preflight,
)
from ml.alphazero_lite.seed604_training_executor import (
    AUTHORIZATION,
    CONFIG,
    FROZEN,
    MEMBERSHIP,
    PARENT,
    RUNTIME,
    TRAINABLE,
    load_freeze,
    verify_authorization,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_artifacts(directory: Path) -> None:
    expected = {
        "b512-reproduction-report.json",
        "training-trace.jsonl",
        "seed604-B2048-update-2048.npz",
        "training-receipt.json",
    }
    actual = {path.name for path in directory.iterdir()}
    if actual != expected:
        raise ValueError("training_output_artifact_set_mismatch")


def _verify_npz(path: Path, parent_path: Path) -> tuple[list[str], list[str]]:
    # Public checkpoint layout is independently checked without importing PyTorch.
    required = {
        "w_input",
        "b_input",
        "w_residual_1_1",
        "b_residual_1_1",
        "w_residual_1_2",
        "b_residual_1_2",
        "w_residual_2_1",
        "b_residual_2_1",
        "w_residual_2_2",
        "b_residual_2_2",
        "w_residual_3_1",
        "b_residual_3_1",
        "w_residual_3_2",
        "b_residual_3_2",
        "w_policy_hidden",
        "b_policy_hidden",
        "w_policy",
        "b_policy",
        "w_value_hidden",
        "b_value_hidden",
        "w_value",
        "b_value",
    }
    with (
        np.load(path, allow_pickle=False) as checkpoint,
        np.load(parent_path, allow_pickle=False) as parent,
    ):
        if set(checkpoint.files) != required or set(parent.files) != required:
            raise ValueError("checkpoint_tensor_names_mismatch")
        for name in sorted(required):
            value = checkpoint[name]
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"checkpoint_tensor_invalid:{name}")
            if value.shape != parent[name].shape:
                raise ValueError(f"checkpoint_tensor_shape_mismatch:{name}")
        frozen_keys = ["w_value_hidden", "b_value_hidden", "w_value", "b_value"]
        changed = [
            key
            for key in frozen_keys
            if not np.array_equal(checkpoint[key], parent[key])
        ]
        if changed:
            raise ValueError(f"frozen_checkpoint_tensor_changed:{changed[0]}")
    return sorted(required), sorted(frozen_keys)


def verify_result(output: Path, *, root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    expected = (root / DATA / "training-run-B2048").resolve()
    if output != expected:
        raise ValueError("noncanonical_output_directory")
    freeze, freeze_raw = load_freeze(root)
    authorization = verify_authorization(root)
    if (
        digest(root / AUTHORIZATION)
        != hashlib.sha256((root / AUTHORIZATION).read_bytes()).hexdigest()
    ):
        raise ValueError("authorization_identity_invalid")
    preflight = read_only_preflight(root)
    rows = json.loads((root / MEMBERSHIP).read_bytes())
    config = json.loads((root / CONFIG).read_bytes())
    schedule = batch_schedule()
    if hashlib.sha256(canonical_bytes(rows)).hexdigest() != config["membership_sha256"]:
        raise ValueError("membership_identity_mismatch")
    if len(schedule) != UPDATES or schedule[:512] != json.loads(
        (root / (SEED599 / "batch-schedule-B.json")).read_bytes()
    ):
        raise ValueError("schedule_identity_mismatch")
    registered_schedule_sha = json.loads(
        (root / (DATA / "registration.json")).read_bytes()
    )["treatment"]["batch_schedule_sha256"]
    if hashlib.sha256(canonical_bytes(schedule)).hexdigest() != registered_schedule_sha:
        raise ValueError("schedule_identity_mismatch")
    parent_sha = preflight["authenticated_seed599_inputs"]["parent_checkpoint_sha256"]
    if digest(root / PARENT) != parent_sha:
        raise ValueError("parent_checkpoint_identity_mismatch")
    _require_artifacts(output)
    receipt_path = output / "training-receipt.json"
    trace_path = output / "training-trace.jsonl"
    checkpoint_path = output / "seed604-B2048-update-2048.npz"
    report_path = output / "b512-reproduction-report.json"
    receipt = json.loads(receipt_path.read_bytes())
    report = json.loads(report_path.read_bytes())
    if (
        receipt.get("schema") != "seed604-b2048-training-result-v1"
        or receipt.get("updates") != UPDATES
    ):
        raise ValueError("training_receipt_identity_mismatch")
    if receipt.get("execution_freeze_sha256") != hashlib.sha256(freeze_raw).hexdigest():
        raise ValueError("training_receipt_freeze_mismatch")
    if receipt.get("authorization_sha256") != digest(root / AUTHORIZATION):
        raise ValueError("training_receipt_authorization_mismatch")
    if receipt.get("b512_report") != report_path.name or receipt.get(
        "b512_report_sha256"
    ) != digest(report_path):
        raise ValueError("training_b512_report_hash_mismatch")
    if receipt.get("membership_sha256") != config["membership_sha256"]:
        raise ValueError("training_receipt_membership_mismatch")
    if receipt.get("registration_sha256") != digest(
        root / (DATA / "registration.json")
    ):
        raise ValueError("training_receipt_registration_mismatch")
    schedule_sha = hashlib.sha256(canonical_bytes(schedule)).hexdigest()
    if receipt.get("schedule_sha256") != schedule_sha:
        raise ValueError("training_receipt_schedule_mismatch")
    if receipt.get("runtime") != RUNTIME or freeze.get("runtime") != RUNTIME:
        raise ValueError("training_runtime_identity_mismatch")
    if (
        receipt.get("b512_reproduction") != "exact-tensor-equality-passed"
        or report.get("passed") is not True
    ):
        raise ValueError("b512_reproduction_evidence_missing")
    comparisons = report.get("comparisons")
    if (
        not isinstance(comparisons, list)
        or not comparisons
        or any(item.get("equal") is not True for item in comparisons)
    ):
        raise ValueError("b512_tensor_comparison_incomplete")
    expected_names = {
        "input_layer.weight",
        "input_layer.bias",
        "policy_hidden_layer.weight",
        "policy_hidden_layer.bias",
        "policy_head.weight",
        "policy_head.bias",
        "value_hidden_layer.weight",
        "value_hidden_layer.bias",
        "value_head.weight",
        "value_head.bias",
        *(
            f"residual_layers.{block}.{layer}.{field}"
            for block in range(3)
            for layer in range(2)
            for field in ("weight", "bias")
        ),
    }
    if (
        report.get("update") != 512
        or {item.get("name") for item in comparisons} != expected_names
        or report.get("tensor_count") != len(expected_names)
        or report.get("parent_checkpoint_sha256") != parent_sha
        or report.get("b512_checkpoint_sha256")
        != preflight["authenticated_seed599_inputs"]["b512_checkpoint_sha256"]
        or report.get("membership_sha256") != config["membership_sha256"]
        or report.get("schedule_sha256") != schedule_sha
    ):
        raise ValueError("b512_tensor_comparison_identity_mismatch")
    expected_shapes = {
        "input_layer.weight": [96, 27],
        "input_layer.bias": [96],
        "policy_hidden_layer.weight": [96, 96],
        "policy_hidden_layer.bias": [96],
        "policy_head.weight": [6, 96],
        "policy_head.bias": [6],
        "value_hidden_layer.weight": [48, 96],
        "value_hidden_layer.bias": [48],
        "value_head.weight": [1, 48],
        "value_head.bias": [1],
    }
    for block in range(3):
        for layer in range(2):
            expected_shapes[f"residual_layers.{block}.{layer}.weight"] = [96, 96]
            expected_shapes[f"residual_layers.{block}.{layer}.bias"] = [96]
    if any(
        item.get("shape") != expected_shapes[item["name"]]
        or item.get("dtype") != "torch.float32"
        for item in comparisons
    ):
        raise ValueError("b512_tensor_schema_evidence_mismatch")
    trainable_names = sorted(
        name for name in expected_names if name.startswith(TRAINABLE)
    )
    optimizer_evidence = report.get("optimizer_state")
    if not isinstance(optimizer_evidence, dict):
        raise ValueError("b512_optimizer_state_evidence_missing")
    state_steps = optimizer_evidence.get("step_counters")
    if (
        optimizer_evidence.get("parameter_count") != len(trainable_names)
        or not isinstance(state_steps, list)
        or [item.get("parameter") for item in state_steps] != trainable_names
        or any(item.get("step") != 512 for item in state_steps)
        or not re.fullmatch(r"[0-9a-f]{64}", optimizer_evidence.get("state_sha256", ""))
        or report.get("optimizer_identity") != "same-live-optimizer-through-update-513"
    ):
        raise ValueError("b512_optimizer_state_evidence_invalid")
    continuity = report.get("optimizer_continuity")
    if not isinstance(continuity, dict) or continuity.get("same_object_id") is not True:
        raise ValueError("optimizer_continuity_evidence_missing")
    before = continuity.get("before_update_513")
    after = continuity.get("after_update_513")
    if (
        before != optimizer_evidence
        or not isinstance(after, dict)
        or after.get("parameter_count") != len(trainable_names)
        or [item.get("parameter") for item in after.get("step_counters", [])]
        != trainable_names
        or any(item.get("step") != 513 for item in after.get("step_counters", []))
        or not re.fullmatch(r"[0-9a-f]{64}", after.get("state_sha256", ""))
    ):
        raise ValueError("optimizer_continuity_evidence_invalid")
    raw_lines = trace_path.read_bytes().splitlines()
    if len(raw_lines) != UPDATES:
        raise ValueError("training_trace_count_mismatch")
    for update, (raw, indexes) in enumerate(zip(raw_lines, schedule, strict=True), 1):
        record = json.loads(raw)
        if record.get("update") != update or record.get("indexes") != indexes:
            raise ValueError(f"training_trace_order_mismatch:{update}")
        if record.get("runtime") != RUNTIME or record.get("seed") != 599:
            raise ValueError(f"training_trace_runtime_mismatch:{update}")
        if not math.isfinite(record.get("loss", math.nan)) or not math.isfinite(
            record.get("gradient_norm", math.nan)
        ):
            raise ValueError(f"training_trace_nonfinite:{update}")
    checkpoint_keys, frozen_keys = _verify_npz(checkpoint_path, root / PARENT)
    for key, path in (
        ("checkpoint_sha256", checkpoint_path),
        ("trace_sha256", trace_path),
    ):
        if receipt.get(key) != digest(path):
            raise ValueError(f"training_artifact_hash_mismatch:{key}")
    if (
        receipt.get("checkpoint") != checkpoint_path.name
        or receipt.get("trace") != trace_path.name
    ):
        raise ValueError("training_artifact_name_mismatch")
    if receipt.get("frozen_parameter_integrity") != {
        name: True for name in receipt.get("frozen_parameter_integrity", {})
    }:
        raise ValueError("frozen_parameter_receipt_invalid")
    expected_frozen = [
        "value_head.bias",
        "value_head.weight",
        "value_hidden_layer.bias",
        "value_hidden_layer.weight",
    ]
    if sorted(receipt.get("frozen_parameter_integrity", {})) != expected_frozen:
        raise ValueError("frozen_parameter_receipt_incomplete")
    # Keep authorization consumed explicitly so its well-formed owner assertion is part of this check.
    if authorization.get("status") != "authorized-for-training":
        raise ValueError("training_authorization_invalid")
    return {
        "passed": True,
        "updates": UPDATES,
        "checkpoint_sha256": digest(checkpoint_path),
        "trace_sha256": digest(trace_path),
        "checkpoint_tensor_names": checkpoint_keys,
        "frozen_checkpoint_tensors": frozen_keys,
        "b512_tensor_comparisons": len(comparisons),
        "training_groups": {
            "trainable_prefixes": list(TRAINABLE),
            "frozen_prefixes": list(FROZEN),
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_result(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
