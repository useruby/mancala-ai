"""Authorization-gated Seed 604 B2048 training executor.

Importing this module is inert. Production training is reachable only through
``train_b2048`` after authenticating the frozen plan and separate owner record.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
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

FREEZE = DATA / "training-execution-freeze.json"
AUTHORIZATION = DATA / "training-authorization.json"
OUTPUT = DATA / "training-run-B2048"
PARENT = Path("docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/model.npz")
B512 = SEED599 / "training-run-B/seed599-arm-B-update-512.npz"
MEMBERSHIP = SEED599 / "treatment-membership.json"
SCHEDULE_512 = SEED599 / "batch-schedule-B.json"
SCHEDULE_2048 = DATA / "batch-schedule-B2048.json"
CONFIG = SEED599 / "training-B.json"
TRAINABLE = ("input_layer.", "residual_layers.", "policy_hidden_layer.", "policy_head.")
FROZEN = ("value_hidden_layer.", "value_head.")
RUNTIME = {
    "python": "3.14.7",
    "numpy": "2.5.2",
    "torch": "2.13.0+cu130",
    "device": "cpu",
    "threads": 1,
    "pythonhashseed": "599",
    "omp_num_threads": "1",
    "mkl_num_threads": "1",
    "deterministic_algorithms": True,
}
FREEZE_SOURCE_PATHS = {
    "ml/alphazero_lite/seed604_training_executor.py",
    "ml/alphazero_lite/verify_seed604_training_result.py",
    "ml/alphazero_lite/seed604_replay_optimization.py",
    "ml/alphazero_lite/seed599_replay_coverage.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/seed604_bounded_exclusions.py",
    "ml/alphazero_lite/verify_seed604_opening_cohort.py",
    "ml/alphazero_lite/seed551_controlled_policy_strength.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_freeze(root: Path = ROOT) -> tuple[dict[str, Any], bytes]:
    raw = (root / FREEZE).read_bytes()
    freeze = json.loads(raw)
    if (
        freeze.get("schema") != "seed604-training-execution-freeze-v1"
        or freeze.get("status") != "frozen-awaiting-owner-authorization"
        or freeze.get("execution_authorized") is not False
    ):
        raise ValueError("execution_freeze_invalid")
    sources = freeze.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != FREEZE_SOURCE_PATHS:
        raise ValueError("execution_source_hash_set_mismatch")
    for relative, expected in sources.items():
        if not isinstance(expected, str) or sha256(root / relative) != expected:
            raise ValueError(f"execution_source_hash_mismatch:{relative}")
    return freeze, raw


def verify_authorization(root: Path = ROOT) -> dict[str, Any]:
    freeze, raw = load_freeze(root)
    record = json.loads((root / AUTHORIZATION).read_bytes())
    owner = record.get("authorization")
    if (
        record.get("schema") != "seed604-training-owner-authorization-v1"
        or record.get("status") != "authorized-for-training"
        or record.get("execution_freeze_sha256") != hashlib.sha256(raw).hexdigest()
        or not isinstance(owner, dict)
        or owner.get("type") != "explicit-project-owner-authorization"
        or not isinstance(owner.get("authorized_by"), str)
        or not owner["authorized_by"].strip()
        or owner.get("training_may_start") is not True
        or freeze.get("execution_authorized") is not False
    ):
        raise ValueError("training_not_authorized_or_freeze_binding_invalid")
    return record


def runtime_identity(torch_module: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch_module.__version__),
        "device": "cpu",
        "threads": torch_module.get_num_threads(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
    }


def check_runtime(expected: dict[str, Any], torch_module: Any) -> None:
    if runtime_identity(torch_module) != expected:
        raise ValueError("runtime_mismatch")
    if torch_module.cuda.is_available():
        # CPU semantics are mandatory even when a GPU is visible.
        pass
    torch_module.set_num_threads(1)
    torch_module.use_deterministic_algorithms(True)


def validate_schedule(
    root: Path = ROOT,
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    read_only_preflight(root)
    rows = json.loads((root / MEMBERSHIP).read_bytes())
    schedule = batch_schedule()
    config = json.loads((root / CONFIG).read_bytes())
    if (
        len(rows) != 2048
        or hashlib.sha256(canonical_bytes(rows)).hexdigest()
        != config["membership_sha256"]
    ):
        raise ValueError("training_membership_identity_mismatch")
    if len(schedule) != UPDATES:
        raise ValueError("training_schedule_identity_mismatch")
    old_schedule = json.loads((root / SCHEDULE_512).read_bytes())
    if schedule[:512] != old_schedule:
        raise ValueError("B512_schedule_prefix_mismatch")
    expected_digest = json.loads((root / (DATA / "registration.json")).read_bytes())[
        "treatment"
    ]["batch_schedule_sha256"]
    if hashlib.sha256(canonical_bytes(schedule)).hexdigest() != expected_digest:
        raise ValueError("training_schedule_identity_mismatch")
    return rows, schedule


def compare_model_tensors(model: Any, checkpoint_path: Path) -> dict[str, Any]:
    """Compare full named tensor sets exactly, including schema and dtype."""
    import torch
    from ml.alphazero_lite import train

    expected_model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    if train.load_checkpoint_into_model(expected_model, checkpoint_path):
        raise ValueError("checkpoint_tensor_load_mismatch")
    actual, expected = dict(model.state_dict()), expected_model.state_dict()
    if set(actual) != set(expected):
        raise ValueError("b512_tensor_names_mismatch")
    comparisons = []
    for name in sorted(actual):
        left, right = actual[name], expected[name]
        same = (
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
        comparisons.append(
            {
                "name": name,
                "shape": list(left.shape),
                "dtype": str(left.dtype),
                "equal": bool(same),
            }
        )
    if not all(item["equal"] for item in comparisons):
        raise ValueError("b512_exact_tensor_mismatch")
    return {
        "passed": True,
        "tensor_count": len(comparisons),
        "comparisons": comparisons,
    }


def _finite_optimizer(optimizer: Any, torch_module: Any) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if (
                isinstance(value, torch_module.Tensor)
                and not torch_module.isfinite(value).all()
            ):
                return False
    return True


def train_b2048(output: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Execute only after all immutable, authorization, and runtime checks pass."""
    expected_output = root / OUTPUT
    if output.expanduser().resolve() != expected_output.resolve():
        raise ValueError("noncanonical_output_directory")
    if output.exists():
        raise ValueError("output_directory_exists")
    freeze, freeze_raw = load_freeze(root)
    verify_authorization(root)
    import torch

    check_runtime(freeze["runtime"], torch)
    preflight = read_only_preflight(root)
    rows, schedule = validate_schedule(root)
    config = json.loads((root / CONFIG).read_bytes())
    if (
        sha256(root / PARENT)
        != preflight["authenticated_seed599_inputs"]["parent_checkpoint_sha256"]
    ):
        raise ValueError("parent_checkpoint_hash_mismatch")
    if (
        sha256(root / B512)
        != preflight["authenticated_seed599_inputs"]["b512_checkpoint_sha256"]
    ):
        raise ValueError("b512_checkpoint_hash_mismatch")

    from ml.alphazero_lite import self_play, train
    from ml.alphazero_lite.kalah_rules import KalahGame

    train.set_seed(599)
    model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    if train.load_checkpoint_into_model(model, root / PARENT):
        raise ValueError("parent_checkpoint_tensor_mismatch")
    trainable, frozen = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE):
            parameter.requires_grad_(True)
            trainable.append(parameter)
        elif name.startswith(FROZEN):
            parameter.requires_grad_(False)
            frozen[name] = parameter.detach().clone()
        else:
            raise ValueError(f"unregistered_parameter_group:{name}")
    if not trainable or not frozen:
        raise ValueError("empty_parameter_group")
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
        if moves != row["legal_moves"]:
            raise ValueError("legal_mask_mismatch")
        legal[index, moves] = True
    optimizer = torch.optim.Adam(
        trainable, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    output.mkdir(parents=False, exist_ok=False)
    trace_path = output / "training-trace.jsonl"
    trace_hash = hashlib.sha256()
    gate_report = None
    with trace_path.open("xb") as trace:
        for update, indexes in enumerate(schedule, 1):
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
                trainable, 1.0, error_if_nonfinite=True
            )
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in trainable
            ):
                raise ValueError(f"nonfinite_gradient:{update}")
            optimizer.step()
            if any(
                not torch.isfinite(p).all() for p in model.parameters()
            ) or not _finite_optimizer(optimizer, torch):
                raise ValueError(f"nonfinite_state:{update}")
            record = (
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
                ).encode()
                + b"\n"
            )
            trace.write(record)
            trace_hash.update(record)
            if update == 512:
                gate_report = compare_model_tensors(model, root / B512)
                gate_report.update(
                    {
                        "update": 512,
                        "parent_checkpoint_sha256": sha256(root / PARENT),
                        "b512_checkpoint_sha256": sha256(root / B512),
                        "membership_sha256": config["membership_sha256"],
                        "schedule_sha256": hashlib.sha256(
                            canonical_bytes(schedule)
                        ).hexdigest(),
                    }
                )
                if any(
                    not torch.equal(
                        dict(model.named_parameters())[name].detach(), value
                    )
                    for name, value in frozen.items()
                ):
                    raise ValueError("frozen_parameter_changed_at_b512")
                gate_report["optimizer_state_entries_retained"] = len(optimizer.state)
                gate_report["optimizer_state_step"] = min(
                    int(state["step"].item())
                    for state in optimizer.state.values()
                    if "step" in state
                )
    if gate_report is None or gate_report.get("passed") is not True:
        raise ValueError("b512_reproduction_gate_not_passed")
    checkpoint = output / "seed604-B2048-update-2048.npz"
    np.savez(checkpoint, **train.checkpoint_from_model(model))
    frozen_integrity = {
        name: torch.equal(dict(model.named_parameters())[name].detach(), value)
        for name, value in frozen.items()
    }
    if not all(frozen_integrity.values()):
        raise ValueError("frozen_parameter_changed")
    (output / "b512-reproduction-report.json").write_text(
        json.dumps(gate_report, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "schema": "seed604-b2048-training-result-v1",
        "execution_freeze_sha256": hashlib.sha256(freeze_raw).hexdigest(),
        "authorization_sha256": sha256(root / AUTHORIZATION),
        "registration_sha256": sha256(root / (DATA / "registration.json")),
        "membership_sha256": config["membership_sha256"],
        "schedule_sha256": hashlib.sha256(canonical_bytes(schedule)).hexdigest(),
        "updates": UPDATES,
        "runtime": freeze["runtime"],
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "trace": trace_path.name,
        "trace_sha256": trace_hash.hexdigest(),
        "b512_report": "b512-reproduction-report.json",
        "b512_report_sha256": sha256(output / "b512-reproduction-report.json"),
        "b512_reproduction": "exact-tensor-equality-passed",
        "frozen_parameter_integrity": frozen_integrity,
    }
    (output / "training-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(train_b2048(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
