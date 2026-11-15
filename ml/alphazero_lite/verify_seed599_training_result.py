"""Independent structural verification for one Seed 599 training result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.seed599_replay_coverage import ROOT, canonical_bytes
from ml.alphazero_lite.seed599_training_executor import (
    FROZEN,
    PARENT,
    PARENT_SHA256,
    TRAINABLE,
    load_freeze,
)

DATA = ROOT / "docs/data/seed599-replay-coverage"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_result(arm: str, output: Path, *, root: Path = ROOT) -> dict[str, Any]:
    if arm not in ("A", "B"):
        raise ValueError("invalid_training_arm")
    freeze, _ = load_freeze(root)
    data = root / DATA.relative_to(ROOT)
    config = json.loads((data / f"training-{arm}.json").read_bytes())
    if (
        digest(data / f"training-{arm}.json")
        != freeze["arms"][arm]["configuration_sha256"]
    ):
        raise ValueError("training_configuration_identity_mismatch")
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
    schedule = json.loads((data / f"batch-schedule-{arm}.json").read_bytes())
    if (
        len(schedule) != 512
        or hashlib.sha256(canonical_bytes(schedule)).hexdigest()
        != config["batch_schedule_sha256"]
    ):
        raise ValueError("training_schedule_identity_mismatch")
    receipt_path = output / "training-receipt.json"
    trace_path = output / "training-trace.jsonl"
    checkpoint_name = f"seed599-arm-{arm}-update-512.npz"
    checkpoint = output / checkpoint_name
    for artifact in (receipt_path, trace_path, checkpoint):
        if not artifact.is_file():
            raise ValueError(f"incomplete_training_output:{artifact.name}")
    receipt = json.loads(receipt_path.read_bytes())
    expected_identity = hashlib.sha256(
        (
            root / "docs/data/seed599-replay-coverage/training-execution-freeze.json"
        ).read_bytes()
    ).hexdigest()
    if (
        receipt.get("schema") != "seed599-training-result-v1"
        or receipt.get("arm") != arm
    ):
        raise ValueError("training_receipt_identity_mismatch")
    if receipt.get("states") != config["states"] or receipt.get("updates") != 512:
        raise ValueError("training_receipt_schedule_mismatch")
    if (
        receipt.get("membership_sha256") != config["membership_sha256"]
        or receipt.get("batch_schedule_sha256") != config["batch_schedule_sha256"]
        or receipt.get("parent_checkpoint_sha256") != PARENT_SHA256
    ):
        raise ValueError("training_receipt_input_identity_mismatch")
    if receipt.get("execution_freeze_sha256") != expected_identity:
        raise ValueError("training_receipt_freeze_mismatch")
    for key, path in (("checkpoint_sha256", checkpoint), ("trace_sha256", trace_path)):
        if receipt.get(key) != digest(path):
            raise ValueError(f"training_artifact_hash_mismatch:{key}")
    if (
        receipt.get("checkpoint") != checkpoint_name
        or receipt.get("trace") != trace_path.name
    ):
        raise ValueError("training_artifact_name_mismatch")
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    if len(trace) != 512:
        raise ValueError("training_trace_update_count_mismatch")
    for update, (record, indexes) in enumerate(zip(trace, schedule, strict=True), 1):
        if record.get("update") != update or record.get("indexes") != indexes:
            raise ValueError(f"training_trace_schedule_mismatch:{update}")
        if record.get("seed") != 599 or record.get("runtime") != freeze["runtime"]:
            raise ValueError(f"training_trace_runtime_mismatch:{update}")
        if not np.isfinite(record.get("loss", np.nan)) or not np.isfinite(
            record.get("gradient_norm", np.nan)
        ):
            raise ValueError(f"training_trace_nonfinite:{update}")
    if digest(root / PARENT.relative_to(ROOT)) != PARENT_SHA256:
        raise ValueError("parent_checkpoint_hash_mismatch")
    parent_model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    result_model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    if train.load_checkpoint_into_model(parent_model, root / PARENT.relative_to(ROOT)):
        raise ValueError("parent_checkpoint_tensor_identity_mismatch")
    if train.load_checkpoint_into_model(result_model, checkpoint):
        raise ValueError("checkpoint_tensor_identity_mismatch")
    names = dict(result_model.named_parameters())
    parent_names = dict(parent_model.named_parameters())
    groups = {"trainable": [], "frozen": []}
    for name, parameter in names.items():
        if name.startswith(TRAINABLE):
            groups["trainable"].append(name)
        elif name.startswith(FROZEN):
            groups["frozen"].append(name)
            if not torch.equal(parameter.detach(), parent_names[name].detach()):
                raise ValueError(f"frozen_parameter_changed:{name}")
        else:
            raise ValueError(f"unregistered_parameter_group:{name}")
    if not groups["trainable"] or not groups["frozen"]:
        raise ValueError("empty_parameter_group")
    if (
        receipt.get("frozen_parameter_names") != sorted(groups["frozen"])
        or receipt.get("frozen_parameters_unchanged") is not True
    ):
        raise ValueError("frozen_parameter_receipt_mismatch")
    return {
        "passed": True,
        "arm": arm,
        "states": len(rows),
        "updates": len(trace),
        "checkpoint_sha256": digest(checkpoint),
        "trace_sha256": digest(trace_path),
        "parameter_groups": groups,
        "frozen_parameter_integrity": "receipt asserted; runtime snapshot is produced by executor",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("A", "B"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(verify_result(args.arm, args.output_dir), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
