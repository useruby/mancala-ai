"""Authorization-gated deterministic Seed 615 training executor.

Importing this module is inert. Training requires a future owner record bound
to the exact published execution-freeze bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed599_replay_coverage as seed599
from ml.alphazero_lite import seed615_opening_diversity as seed615

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed615-source-opening-diversity")
SEED599 = Path("docs/data/seed599-replay-coverage")
FREEZE = DATA / "training-execution-freeze-v3.json"
AUTHORIZATION = DATA / "training-authorization.json"
OUTPUT = DATA / "training-run-S615"
RUNTIME_CERTIFICATION = DATA / "training-runtime-certification-v3.json"
SUPERSEDED_FREEZE = DATA / "training-execution-freeze.json"
SUPERSEDED_READINESS = DATA / "training-readiness-evidence.json"
SUPERSEDED_V2_FREEZE = DATA / "training-execution-freeze-v2.json"
SUPERSEDED_V2_READINESS = DATA / "training-readiness-evidence-v2.json"
REGISTRATION = DATA / "registration.json"
MEMBERSHIP = DATA / "candidate-membership.jsonl"
SCHEDULE = SEED599 / "batch-schedule-B.json"
CONFIG = SEED599 / "training-B.json"
PARENT = Path("docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/model.npz")
B512 = SEED599 / "training-run-B/seed599-arm-B-update-512.npz"
PARENT_SHA256 = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
B512_SHA256 = "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c"
MEMBERSHIP_SHA256 = "5ed236266ed3804a9c0e9f75b7e10b7c81ea6e290f8c72d3cea06dd225a15010"
SCHEDULE_SHA256 = "62edb14805d20af1cc7988d73aba95dcdf9e67842b6ea1d61a12c98a87c28599"
UPDATES = 512
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
SOURCE_PATHS = {
    "ml/alphazero_lite/seed615_training_executor.py",
    "ml/alphazero_lite/verify_seed615_training_result.py",
    "ml/alphazero_lite/verify_seed615_training_freeze.py",
    "ml/alphazero_lite/test_seed615_training_executor.py",
    "ml/alphazero_lite/seed615_opening_diversity.py",
    "ml/alphazero_lite/verify_seed615_opening_diversity.py",
    "ml/alphazero_lite/seed599_replay_coverage.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/requirements-seed604-training.txt",
    "ml/alphazero_lite/verify_seed604_runtime_certification.py",
}
EXECUTION_POLICY = (
    "Validate all inputs and exact owner authorization before creating output; "
    "one attempt only; no automatic retry or recovery; preserve partial output "
    "for forensic review."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def membership_rows(root: Path = ROOT) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (root / MEMBERSHIP).read_text().splitlines()
        if line
    ]
    if (
        len(rows) != 2048
        or hashlib.sha256(canonical_bytes(rows)).hexdigest() != MEMBERSHIP_SHA256
    ):
        raise ValueError("membership_identity_mismatch")
    if rows != sorted(rows, key=lambda row: (row["phase"], row["canonical_identity"])):
        raise ValueError("membership_order_mismatch")
    quotas = {"above_32": 512, "17_to_32": 1024, "at_most_16": 512}
    if {
        phase: sum(row["phase"] == phase for row in rows) for phase in quotas
    } != quotas:
        raise ValueError("membership_phase_quota_mismatch")
    if len({row["canonical_identity"] for row in rows}) != 2048:
        raise ValueError("membership_canonical_duplicates")
    if len({row["float32_identity"] for row in rows}) != 2048:
        raise ValueError("membership_float32_duplicates")
    for row in rows:
        seed599.validate_target_record(row)
    return rows


def batch_schedule(root: Path = ROOT) -> list[list[int]]:
    schedule = json.loads((root / SCHEDULE).read_bytes())
    digest = hashlib.sha256(canonical_bytes(schedule)).hexdigest()
    if digest != SCHEDULE_SHA256 or len(schedule) != UPDATES:
        raise ValueError("batch_schedule_identity_mismatch")
    for indexes in schedule:
        if (
            len(indexes) != 32
            or len(set(indexes)) != 32
            or any(type(index) is not int or not 0 <= index < 2048 for index in indexes)
        ):
            raise ValueError("invalid_batch_schedule")
    return schedule


def validate_checkpoints(root: Path = ROOT) -> None:
    if sha256(root / PARENT) != PARENT_SHA256:
        raise ValueError("parent_checkpoint_identity_mismatch")
    if sha256(root / B512) != B512_SHA256:
        raise ValueError("b512_checkpoint_identity_mismatch")


def expected_freeze(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    preflight = seed599.read_only_preflight(root)
    seed615.verify(root)
    rows = membership_rows(root)
    schedule = batch_schedule(root)
    validate_checkpoints(root)
    if preflight["training_or_evaluation_started"] is not False:
        raise ValueError("preflight_side_effect_detected")
    registration_sha = sha256(root / REGISTRATION)
    config = json.loads((root / CONFIG).read_bytes())
    if config.get("rng_seed") != 599:
        raise ValueError("historical_rng_semantics_mismatch")
    return {
        "schema": "seed615-training-execution-freeze-v1",
        "status": "frozen-awaiting-owner-authorization",
        "training_authorized": False,
        "bindings": {
            "registration_sha256": registration_sha,
            "parent_checkpoint_sha256": PARENT_SHA256,
            "b512_checkpoint_sha256": B512_SHA256,
            "membership_sha256": MEMBERSHIP_SHA256,
            "batch_schedule_sha256": SCHEDULE_SHA256,
            "heldout_provenance_sha256": seed615.HELDOUT_SHA256,
            "seed599_training_configuration_sha256": sha256(root / CONFIG),
            "runtime_certification_sha256": sha256(root / RUNTIME_CERTIFICATION),
            "superseded_freeze_sha256": sha256(root / SUPERSEDED_FREEZE),
            "superseded_readiness_sha256": sha256(root / SUPERSEDED_READINESS),
            "superseded_v2_freeze_sha256": sha256(root / SUPERSEDED_V2_FREEZE),
            "superseded_v2_readiness_sha256": sha256(root / SUPERSEDED_V2_READINESS),
            "superseded_v2_runtime_certification_sha256": sha256(
                root / DATA / "training-runtime-certification-v2.json"
            ),
            "schedule_rows": len(schedule),
            "membership_rows": len(rows),
        },
        "architecture": {
            "model_type": "residual_v3",
            "input_size": 27,
            "hidden_sizes": [96, 3],
        },
        "parameter_groups": {"trainable": list(TRAINABLE), "frozen": list(FROZEN)},
        "optimizer": {
            "name": "Adam",
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "scheduler": None,
        },
        "training": {
            "updates": 512,
            "batch_size": 32,
            "gradient_clip_norm": 1.0,
            "loss": "masked_policy_cross_entropy",
            "device": "cpu",
            "torch_threads": 1,
        },
        "rng": {
            "training_seed": 599,
            "semantics": "Seed 599 train.set_seed(599) before architecture construction; then load exact P-E4 state and use fresh Adam; frozen Seed 599 B schedule indexes sorted Seed 615 membership rows; no per-update randomness.",
        },
        "runtime": dict(RUNTIME),
        "hardware_requirements": {
            "architecture": "x86_64",
            "device": "CPU only",
            "threads": 1,
            "cuda_execution": False,
        },
        "source_sha256": {path: sha256(root / path) for path in sorted(SOURCE_PATHS)},
        "dependency_lock": "docs/data/seed604-replay-optimization/runtime-certification-v3/requirements-lock.txt",
        "dependency_lock_sha256": sha256(
            root
            / "docs/data/seed604-replay-optimization/runtime-certification-v3/requirements-lock.txt"
        ),
        "output_contract": {
            "directory": str(OUTPUT),
            "create_only": True,
            "updates": 512,
            "result_schema": "seed615-training-result-v1",
            "artifacts": [
                "training-trace.jsonl",
                "seed615-S615-update-512.npz",
                "training-receipt.json",
                "training-integrity-report.json",
            ],
            "final_checkpoint_only": True,
            "intermediate_checkpoint_selection": False,
        },
        "execution_policy": EXECUTION_POLICY,
        "diagnostics": {
            "compare_final_tensors_with_b512": True,
            "report_equality_without_selection": True,
            "report_trainable_movement_from_parent_without_selection": True,
        },
    }


def load_freeze(root: Path = ROOT) -> tuple[dict[str, Any], bytes]:
    raw = (root / FREEZE).read_bytes()
    freeze = json.loads(raw)
    if freeze != expected_freeze(root):
        raise ValueError("execution_freeze_contract_mismatch")
    if raw != canonical_bytes(freeze) + b"\n":
        raise ValueError("execution_freeze_not_canonical")
    return freeze, raw


def verify_authorization(root: Path = ROOT) -> dict[str, Any]:
    _freeze, raw = load_freeze(root)
    try:
        record = json.loads((root / AUTHORIZATION).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("training_authorization_missing_or_malformed") from error
    owner = record.get("authorization", {})
    if (
        record.get("schema") != "seed615-training-owner-authorization-v1"
        or record.get("status") != "authorized-for-training"
        or record.get("execution_freeze_sha256") != hashlib.sha256(raw).hexdigest()
        or owner.get("type") != "explicit-project-owner-authorization"
        or not isinstance(owner.get("authorized_by"), str)
        or not owner["authorized_by"].strip()
        or owner.get("training_may_start") is not True
    ):
        raise ValueError("training_authorization_or_freeze_binding_invalid")
    return record


def runtime_identity(torch: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def check_runtime(torch: Any) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if runtime_identity(torch) != RUNTIME:
        raise ValueError("runtime_identity_mismatch")


def train_s615(output: Path, *, root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    if output.expanduser().resolve() != (root / OUTPUT).resolve():
        raise ValueError("noncanonical_output_directory")
    if output.exists():
        raise ValueError("output_directory_exists")
    freeze, freeze_raw = load_freeze(root)
    authorization = verify_authorization(root)
    import torch

    check_runtime(torch)
    seed599.read_only_preflight(root)
    rows = membership_rows(root)
    schedule = batch_schedule(root)
    validate_checkpoints(root)

    from ml.alphazero_lite import self_play, train
    from ml.alphazero_lite.kalah_rules import KalahGame

    train.set_seed(599)
    model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    if train.load_checkpoint_into_model(model, root / PARENT):
        raise ValueError("parent_checkpoint_tensor_mismatch")
    named = dict(model.named_parameters())
    trainable, named_trainable, frozen = classify_parameters(model)
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
    validate_optimizer(optimizer, trainable, torch)
    # All registered inputs and runtime are validated before the create-only output.
    output.mkdir(parents=False, exist_ok=False)
    trace_path = output / "training-trace.jsonl"
    trace_hash = hashlib.sha256()
    with trace_path.open("xb") as trace:
        for update, indexes in enumerate(schedule, 1):
            index = torch.tensor(indexes, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits, _value = model(x[index])
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
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in trainable
            ):
                raise ValueError(f"nonfinite_gradient:{update}")
            optimizer.step()
            validate_finite_state(model, optimizer, torch, update)
            record = {
                "update": update,
                "indexes": indexes,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "seed": 599,
                "runtime": freeze["runtime"],
            }
            raw = (
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )
            trace.write(raw)
            trace_hash.update(raw)
    state_evidence = _verify_adam(optimizer, named_trainable, torch)
    validate_frozen_parameters(named, frozen, torch)
    checkpoint = output / "seed615-S615-update-512.npz"
    np.savez(checkpoint, **train.checkpoint_from_model(model))
    final_arrays = train.checkpoint_from_model(model)
    with (
        np.load(root / PARENT, allow_pickle=False) as initial_checkpoint,
        np.load(root / B512, allow_pickle=False) as control_checkpoint,
    ):
        diagnostics = checkpoint_diagnostics(
            final_arrays,
            dict(initial_checkpoint.items()),
            dict(control_checkpoint.items()),
            named_trainable,
        )
    report = {
        "schema": "seed615-training-integrity-report-v1",
        "updates": UPDATES,
        "adam_state": state_evidence,
        "frozen_parameters_unchanged": True,
        **diagnostics,
    }
    report_path = output / "training-integrity-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "seed615-training-result-v1",
        "execution_freeze_sha256": hashlib.sha256(freeze_raw).hexdigest(),
        "authorization_sha256": sha256(root / AUTHORIZATION),
        "registration_sha256": sha256(root / REGISTRATION),
        "parent_checkpoint_sha256": PARENT_SHA256,
        "b512_checkpoint_sha256": B512_SHA256,
        "membership_sha256": MEMBERSHIP_SHA256,
        "batch_schedule_sha256": SCHEDULE_SHA256,
        "updates": UPDATES,
        "runtime": freeze["runtime"],
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "trace": trace_path.name,
        "trace_sha256": trace_hash.hexdigest(),
        "integrity_report": report_path.name,
        "integrity_report_sha256": sha256(report_path),
        "frozen_parameters_unchanged": True,
        "owner_authorization_sha256": hashlib.sha256(
            canonical_bytes(authorization)
        ).hexdigest(),
    }
    (output / "training-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def _tensor_key(name: str) -> str:
    keys = {
        "input_layer.weight": "w_input",
        "input_layer.bias": "b_input",
        "policy_hidden_layer.weight": "w_policy_hidden",
        "policy_hidden_layer.bias": "b_policy_hidden",
        "policy_head.weight": "w_policy",
        "policy_head.bias": "b_policy",
    }
    if name in keys:
        return keys[name]
    _, block, layer, field = name.split(".")
    return (
        f"w_residual_{int(block) + 1}_{int(layer) + 1}"
        if field == "weight"
        else f"b_residual_{int(block) + 1}_{int(layer) + 1}"
    )


def checkpoint_diagnostics(
    final: dict[str, Any],
    parent: dict[str, Any],
    b512: dict[str, Any],
    named_trainable: dict[str, Any],
) -> dict[str, Any]:
    """Report tensor equality and movement diagnostically, never as a run gate."""
    if set(final) != set(parent) or set(final) != set(b512):
        raise ValueError("diagnostic_checkpoint_tensor_names_mismatch")
    if any(
        final[name].shape != parent[name].shape or final[name].shape != b512[name].shape
        for name in final
    ):
        raise ValueError("diagnostic_checkpoint_tensor_shapes_mismatch")
    equality = {
        name: bool(np.array_equal(final[name], b512[name])) for name in sorted(final)
    }
    movement = {
        name: bool(
            not np.array_equal(final[_tensor_key(name)], parent[_tensor_key(name)])
        )
        for name in sorted(named_trainable)
    }
    return {
        "trainable_parameter_movement_diagnostic": movement,
        "all_trainable_parameters_moved": all(movement.values()),
        "b512_tensor_equality_diagnostic": equality,
        "b512_equal_tensor_names": [name for name, equal in equality.items() if equal],
        "b512_exact_tensor_equality": all(equality.values()),
        "b512_equality_used_for_selection": False,
    }


def _finite(value: Any, torch: Any) -> bool:
    return not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all())


def _verify_adam(
    optimizer: Any, named_trainable: dict[str, Any], torch: Any
) -> dict[str, Any]:
    expected = set(named_trainable.values())
    if (
        len(optimizer.param_groups) != 1
        or set(optimizer.param_groups[0]["params"]) != expected
        or set(optimizer.state) != expected
    ):
        raise ValueError("adam_state_parameter_set_mismatch")
    digest = hashlib.sha256()
    steps = []
    for name, parameter in sorted(named_trainable.items()):
        state = optimizer.state[parameter]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"adam_state_incomplete:{name}")
        step = float(state["step"].item())
        if not math.isfinite(step) or step != UPDATES:
            raise ValueError(f"adam_step_mismatch:{name}")
        for key in ("exp_avg", "exp_avg_sq"):
            slot = state[key]
            if (
                slot.shape != parameter.shape
                or slot.dtype != parameter.dtype
                or not torch.isfinite(slot).all()
            ):
                raise ValueError(f"adam_slot_invalid:{name}:{key}")
            digest.update(
                name.encode()
                + key.encode()
                + slot.detach().cpu().contiguous().numpy().tobytes()
            )
        steps.append({"parameter": name, "step": int(step)})
    return {
        "parameter_count": len(steps),
        "step_counters": steps,
        "state_sha256": digest.hexdigest(),
    }


def validate_optimizer(optimizer: Any, trainable: list[Any], torch: Any) -> None:
    if not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 1:
        raise ValueError("optimizer_type_or_group_mismatch")
    group = optimizer.param_groups[0]
    if (
        group.get("lr") != 1e-5
        or group.get("betas") != (0.9, 0.999)
        or group.get("eps") != 1e-8
        or group.get("weight_decay") != 0.0
        or len(group.get("params", [])) != len(trainable)
        or any(
            left is not right
            for left, right in zip(group["params"], trainable, strict=True)
        )
    ):
        raise ValueError("optimizer_hyperparameter_or_parameter_mismatch")


def classify_parameters(model: Any) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    trainable: list[Any] = []
    named_trainable: dict[str, Any] = {}
    frozen: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE):
            parameter.requires_grad_(True)
            trainable.append(parameter)
            named_trainable[name] = parameter
        elif name.startswith(FROZEN):
            parameter.requires_grad_(False)
            frozen[name] = parameter.detach().clone()
        else:
            raise ValueError(f"unregistered_parameter_group:{name}")
    if not trainable or not frozen:
        raise ValueError("empty_parameter_group")
    return trainable, named_trainable, frozen


def validate_frozen_parameters(
    named: dict[str, Any], frozen: dict[str, Any], torch: Any
) -> None:
    if any(
        name not in named or not torch.equal(named[name].detach(), value)
        for name, value in frozen.items()
    ):
        raise ValueError("frozen_parameter_changed")


def validate_finite_state(model: Any, optimizer: Any, torch: Any, update: int) -> None:
    if any(
        not torch.isfinite(parameter).all() for parameter in model.parameters()
    ) or any(
        not _finite(value, torch)
        for state in optimizer.state.values()
        for value in state.values()
    ):
        raise ValueError(f"nonfinite_model_or_optimizer_state:{update}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(train_s615(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
