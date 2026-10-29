"""Prospective current-runtime matched-control seed542 C/D comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import (
    publish_seed542_fixed_update_cd_v2 as publication,
    run_seed504,
    run_seed542_fixed_update_cd as fixed_update,
    train,
    verify_seed542_amended_corpus,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
OUT = ROOT / "docs/data/seed542-fixed-update-cd-v2"
TABLEBASE = Path("/home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb")
BOUNDARIES = {27, 54, 81, 108}
ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def array_record(value: np.ndarray | torch.Tensor) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = value
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "strides_bytes": list(array.strides),
        "contiguous": bool(array.flags.c_contiguous),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def named_parameter_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b"\0")
        h.update(str(array.dtype).encode() + b"\0")
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


def optimizer_record(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> dict[str, Any]:
    order = list(model.named_parameters())
    state = {}
    for name, parameter in order:
        values = optimizer.state.get(parameter, {})
        state[name] = {
            key: array_record(value) if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }
    return {
        "parameter_order": [name for name, _parameter in order],
        "groups": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
        "state": state,
    }


def environment_settings() -> dict[str, Any]:
    try:
        cgroup = Path("/proc/self/cgroup").read_text().strip()
    except OSError:
        cgroup = None
    return {
        "interpreter": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "installed_packages": sorted(
            (
                f"{distribution.metadata['Name']}=={distribution.version}"
                for distribution in importlib.metadata.distributions()
                if distribution.metadata.get("Name")
                and distribution.metadata["Name"].lower() != "pip"
            ),
            key=str.casefold,
        ),
        "device": "cpu",
        "dtype": "torch.float32 / numpy.float32",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "environment_variables": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_available": torch.cuda.is_available(),
        "cgroup": cgroup,
        "nice": os.nice(0),
        "resource_limits": {
            "requested_cpu_quota": os.environ.get("SEED542_CD_CPU_QUOTA"),
            "requested_cpu_weight": os.environ.get("SEED542_CD_CPU_WEIGHT"),
            "requested_memory_max": os.environ.get("SEED542_CD_MEMORY_MAX"),
            "requested_allowed_cpus": os.environ.get("SEED542_CD_ALLOWED_CPUS"),
            "requested_nice": os.environ.get("SEED542_CD_NICE"),
            "requested_address_space_limit": os.environ.get("SEED542_CD_ADDRESS_SPACE"),
            "cpu_max": _cgroup_value("cpu.max"),
            "cpu_weight": _cgroup_value("cpu.weight"),
            "memory_max": _cgroup_value("memory.max"),
            "cpuset_effective": _cgroup_value("cpuset.cpus.effective"),
            "address_space_rlimit": list(resource.getrlimit(resource.RLIMIT_AS)),
        },
        "torch_build_config": torch.__config__.show(),
    }


def _cgroup_value(name: str) -> str | None:
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
        v2 = next(
            line.split("::", 1)[1] for line in memberships if line.startswith("0::")
        )
        path = Path("/sys/fs/cgroup") / v2.lstrip("/") / name
        return path.read_text().strip()
    except OSError:
        return None


def source_and_input_hashes() -> tuple[dict[str, str], dict[str, str]]:
    seed504 = json.loads((ROOT / "docs/data/seed504/manifest.json").read_text())
    expanded = json.loads((CORPUS / "source-manifest.json").read_text())
    input_paths = {
        "docs/data/seed542-selfplay-4096/continuation-amendment.json",
        "docs/data/seed542-selfplay-4096/amended-accounting.json",
        "docs/data/seed542-selfplay-4096/source-manifest.json",
        "docs/data/seed542-selfplay-4096/development-membership.json",
        "docs/data/seed542-selfplay-4096/training-protocol.json",
        "docs/data/seed542-selfplay-4096/initializer.npz",
        "docs/data/seed542-selfplay-4096/recovered-seed542001-complete.jsonl",
        "docs/data/seed542-selfplay-4096/recovered-seed542001-provenance.jsonl",
        "docs/data/seed504/manifest.json",
        "docs/data/seed504/permutations.json",
        *(
            source["path"]
            for source in [
                *seed504["sources"]["training"],
                *seed504["sources"]["development"],
                *expanded["sources"],
            ]
        ),
        *(f"docs/data/seed504/seed504_epoch{epoch}.npz" for epoch in range(1, 5)),
    }
    for relative_directory in (
        "docs/data/seed542-fixed-update-cd",
        "docs/data/seed542-fixed-update-cd-reproduction",
        "docs/data/seed542-fixed-update-cd-diagnostics",
    ):
        input_paths.update(
            str(path.relative_to(ROOT))
            for path in (ROOT / relative_directory).rglob("*")
            if path.is_file()
        )
    input_hashes = {relative: sha(ROOT / relative) for relative in sorted(input_paths)}
    source_paths = (
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/checkpoint_phase_selection.py",
        "ml/alphazero_lite/input_encodings.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/exact_root_policy_targets.py",
        "ml/alphazero_lite/run_seed504.py",
        "ml/alphazero_lite/verify_seed542_amended_corpus.py",
        "ml/alphazero_lite/run_seed542_fixed_update_cd.py",
        "ml/alphazero_lite/verify_seed542_fixed_update_cd.py",
        "ml/alphazero_lite/test_seed542_fixed_update_cd.py",
        "ml/alphazero_lite/run_seed542_fixed_update_cd_v2.py",
        "ml/alphazero_lite/verify_seed542_fixed_update_cd_v2.py",
        "ml/alphazero_lite/test_seed542_fixed_update_cd_v2.py",
        "ml/alphazero_lite/publish_seed542_fixed_update_cd_v2.py",
        "ml/alphazero_lite/diagnose_seed542_c_reproduction.py",
        "ml/alphazero_lite/diagnose_seed542_c_first_divergence.py",
        "ml/alphazero_lite/test_seed542_c_reproduction_diagnostic.py",
    )
    return input_hashes, {relative: sha(ROOT / relative) for relative in source_paths}


def freeze_registration(
    out: Path, commands: list[str], tablebase: Path
) -> dict[str, Any]:
    amendment = json.loads((CORPUS / "continuation-amendment.json").read_text())
    input_hashes, sources = source_and_input_hashes()
    runtime = environment_settings()
    if runtime["installed_packages"] != amendment["interpreter"]["installed_packages"]:
        raise ValueError("seed542_v2_runtime_package_set_mismatch")
    if runtime["python"].split()[0] != amendment["environment"]["python"]:
        raise ValueError("seed542_v2_python_version_mismatch")
    if runtime["platform"] != amendment["environment"]["platform"]:
        raise ValueError("seed542_v2_platform_mismatch")
    if (runtime["numpy"], runtime["torch"]) != (
        amendment["environment"]["numpy"],
        amendment["environment"]["torch"],
    ):
        raise ValueError("seed542_v2_numeric_package_version_mismatch")
    if runtime["device"] != "cpu":
        raise ValueError("seed542_v2_device_mismatch")
    if runtime["torch_num_threads"] != 1 or runtime["torch_num_interop_threads"] != 1:
        raise ValueError("seed542_v2_torch_thread_count_mismatch")
    if runtime["cpu_affinity"] != [30]:
        raise ValueError("seed542_v2_cpu_affinity_mismatch")
    if runtime["environment_variables"] != {key: "1" for key in ENVIRONMENT_KEYS}:
        raise ValueError("seed542_v2_thread_environment_mismatch")
    limits = runtime["resource_limits"]
    if (
        limits["requested_cpu_quota"] != "25%"
        or limits["requested_cpu_weight"] != "1"
        or limits["requested_memory_max"] != "4294967296"
        or limits["requested_allowed_cpus"] != "30"
        or limits["requested_nice"] != "19"
        or limits["requested_address_space_limit"] != "17179869184"
        or limits["cpu_max"] != "25000 100000"
        or limits["memory_max"] != "4294967296"
        or limits["cpuset_effective"] not in {None, "30"}
        or limits["cpu_weight"] != "1"
        or limits["address_space_rlimit"] != [17179869184, 17179869184]
        or runtime["nice"] != 19
    ):
        raise ValueError("seed542_v2_resource_limits_mismatch")
    registration = {
        "status": "prospective frozen before production reference or training",
        "experiment": "seed542 expanded-data fixed-update comparison v2",
        "scope": "current-runtime production control; historical seed504 drift reported descriptively",
        "protocol_sha256": sha(CORPUS / "training-protocol.json"),
        "input_sha256": input_hashes,
        "runtime_assets": {
            "tablebase_path_at_registration": str(tablebase.resolve()),
            "tablebase_sha256": sha(tablebase),
            "tablebase_size_bytes": tablebase.stat().st_size,
        },
        "execution_source_sha256": sources,
        "runtime_and_resource_settings": runtime,
        "commands": commands,
        "training_contract": {
            "C": "seed504 original training rows with published frozen permutations",
            "D": "expanded seed542 rows with published fixed without-replacement schedule",
            "updates_per_arm": 108,
            "exposures_per_arm": 54896,
            "batch_sizes": [n for _ in range(4) for n in ([512] * 26 + [412])],
            "architecture": "residual_v3/kalah_v3; hidden=(96,3); 27 inputs",
            "objective": "legal policy CE + 0.5 * Huber(value, delta=1)",
            "optimizer": {
                "name": "Adam",
                "lr": 0.0001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "scheduler": "none",
            },
            "gradient_clip_norm": 1.0,
            "candidate_endpoint": "update 108 only",
            "current_runtime_control_gate": "production and direct C records/parameter state bitwise equal at every update",
            "historical_seed504_equality": "descriptive only; not a validity gate",
        },
    }
    path = out / "registration.json"
    if path.exists():
        raise FileExistsError(f"v2_registration_exists:{path}")
    path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    return registration


def verify_frozen_registration(out: Path) -> dict[str, Any]:
    path = out / "registration.json"
    registration = json.loads(path.read_text())
    if (
        registration.get("status")
        != "prospective frozen before production reference or training"
    ):
        raise ValueError("seed542_v2_registration_status_invalid")
    input_hashes, source_hashes = source_and_input_hashes()
    if registration["input_sha256"] != input_hashes:
        raise ValueError("seed542_v2_registered_inputs_changed_after_freeze")
    if registration["execution_source_sha256"] != source_hashes:
        raise ValueError("seed542_v2_execution_sources_changed_after_freeze")
    if registration["protocol_sha256"] != sha(CORPUS / "training-protocol.json"):
        raise ValueError("seed542_v2_protocol_changed_after_freeze")
    frozen_runtime = dict(registration["runtime_and_resource_settings"])
    current_runtime = environment_settings()
    # systemd scopes get new cgroup path names; execution limits and affinity
    # must match, while the distinct scope identity is separately recorded.
    frozen_runtime.pop("cgroup", None)
    current_runtime.pop("cgroup", None)
    if frozen_runtime != current_runtime:
        raise ValueError("seed542_v2_runtime_or_resource_settings_changed_after_freeze")
    return registration


class ProductionRecorder:
    def __init__(self, identities: list[str], arrays: tuple[np.ndarray, ...]):
        self.identities = identities
        self.x, self.p, self.v, self.weights = arrays
        self.legal = train.legal_mask_matrix_for_encoded_states(self.x)
        self.model: torch.nn.Module | None = None
        self._attached_model: torch.nn.Module | None = None
        self.records: list[dict[str, Any]] = []
        self.latest_forward: dict[str, Any] | None = None
        self.latest_objective: float | None = None
        self.latest_clip_return: float | None = None
        self._hook = None

    def attach(self, model: torch.nn.Module) -> None:
        self.model = model
        if self._attached_model is not model:
            self._hook = model.register_forward_hook(self._forward_hook)
            self._attached_model = model

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()

    def _forward_hook(self, _module: Any, args: tuple[Any, ...], output: Any) -> None:
        if self.model is None:
            raise RuntimeError("production_recorder_model_missing")
        logits, value = output
        self.latest_forward = {
            "input": array_record(args[0]),
            "policy_logits": array_record(logits),
            "value_prediction": array_record(value),
            "model_training": bool(self.model.training),
        }

    def batch(self, indexes: list[int]) -> dict[str, Any]:
        index_tensor = torch.as_tensor(indexes, dtype=torch.int64)
        ids = [self.identities[index] for index in indexes]
        return {
            "row_ids": ids,
            "batch_size": len(indexes),
            "tensors": {
                "input": array_record(torch.from_numpy(self.x)[index_tensor]),
                "target_policy": array_record(torch.from_numpy(self.p)[index_tensor]),
                "target_value": array_record(torch.from_numpy(self.v)[index_tensor]),
                "policy_weight": array_record(
                    torch.from_numpy(self.weights)[index_tensor]
                ),
                "legal_mask": array_record(torch.from_numpy(self.legal)[index_tensor]),
            },
        }

    def loss_observer(self, event: dict[str, Any]) -> None:
        self.latest_objective = float(event["total_loss_tensor"].detach().cpu().item())

    def step_callback(self, phase: str, context: dict[str, Any]) -> None:
        if self.model is None:
            raise RuntimeError("production_recorder_model_missing")
        if phase == "before":
            indexes = context["batch_indexes"]
            self.records.append(
                {
                    "update": len(self.records) + 1,
                    "batch": self.batch(indexes),
                    "forward": self.latest_forward,
                    "policy_loss": context["policy_loss"],
                    "value_loss": context["value_loss"],
                    "weighted_total_objective": self.latest_objective,
                    "production_logged_gradient_norm": context["gradient_norm"],
                    "raw_gradients": {
                        name: array_record(gradient)
                        for name, gradient in context["raw_gradients"].items()
                    },
                    "optimizer_before": optimizer_record(
                        context["optimizer"], self.model
                    ),
                    "parameter_state_before": fixed_update.state_digest(self.model),
                }
            )
        else:
            record = self.records[-1]
            record["clipped_gradients"] = {
                name: array_record(parameter.grad)
                for name, parameter in self.model.named_parameters()
            }
            record["gradient_norm_preclip"] = self.latest_clip_return
            record["gradient_norm_clip_function"] = self.latest_clip_return
            record["optimizer_after"] = optimizer_record(
                context["optimizer"], self.model
            )
            record["parameter_state_after"] = fixed_update.state_digest(self.model)


class DirectRecorder:
    def __init__(
        self,
        model: train.PolicyValueNet,
        x: np.ndarray,
        p: np.ndarray,
        v: np.ndarray,
        weights: np.ndarray,
        identities: list[str],
    ):
        self.model = model
        self.x, self.p, self.v, self.weights = x, p, v, weights
        self.identities = identities
        self.legal = train.legal_mask_matrix_for_encoded_states(x)
        self.forward: dict[str, Any] | None = None
        self.hook = model.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module: Any, args: tuple[Any, ...], output: Any) -> None:
        logits, value = output
        self.forward = {
            "input": array_record(args[0]),
            "policy_logits": array_record(logits),
            "value_prediction": array_record(value),
            "model_training": bool(self.model.training),
        }

    def update(
        self,
        optimizer: torch.optim.Optimizer,
        row_ids: list[str],
        positions: dict[str, int],
    ) -> dict[str, Any]:
        indices = np.asarray([positions[row_id] for row_id in row_ids], dtype=np.int64)
        batch = {
            "row_ids": row_ids,
            "batch_size": len(row_ids),
            "tensors": {
                "input": array_record(torch.from_numpy(self.x[indices])),
                "target_policy": array_record(torch.from_numpy(self.p[indices])),
                "target_value": array_record(torch.from_numpy(self.v[indices])),
                "policy_weight": array_record(torch.from_numpy(self.weights[indices])),
                "legal_mask": array_record(
                    torch.from_numpy(
                        train.legal_mask_matrix_for_encoded_states(self.x[indices])
                    )
                ),
            },
        }
        before_optimizer = optimizer_record(optimizer, self.model)
        before_parameters = fixed_update.state_digest(self.model)
        raw_gradients: dict[str, Any] = {}
        clipped_gradients: dict[str, Any] = {}
        original_clip = torch.nn.utils.clip_grad_norm_
        original_step = optimizer.step
        clip_norm: float | None = None
        after_optimizer: dict[str, Any] | None = None
        after_parameters: str | None = None

        def record_clip(
            parameters: Any, max_norm: float, *args: Any, **kwargs: Any
        ) -> Any:
            nonlocal clip_norm
            parameter_list = list(parameters)
            raw_gradients.update(
                {
                    name: array_record(parameter.grad)
                    for name, parameter in self.model.named_parameters()
                }
            )
            result = original_clip(parameter_list, max_norm, *args, **kwargs)
            clip_norm = float(result.detach().cpu().item())
            clipped_gradients.update(
                {
                    name: array_record(parameter.grad)
                    for name, parameter in self.model.named_parameters()
                }
            )
            return result

        def record_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal after_optimizer, after_parameters
            result = original_step(*args, **kwargs)
            after_optimizer = optimizer_record(optimizer, self.model)
            after_parameters = fixed_update.state_digest(self.model)
            return result

        torch.nn.utils.clip_grad_norm_ = record_clip
        optimizer.step = record_step  # type: ignore[method-assign]
        try:
            result = fixed_update.update(
                self.model,
                optimizer,
                row_ids,
                positions,
                self.x,
                self.p,
                self.v,
                self.weights,
            )
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
            optimizer.step = original_step  # type: ignore[method-assign]
        return {
            "batch": batch,
            "forward": self.forward,
            "policy_loss": result["policy_loss"],
            "value_loss": result["value_loss"],
            "weighted_total_objective": result["weighted_total_objective"],
            "gradient_norm_preclip": clip_norm,
            "gradient_norm_clip_function": clip_norm,
            "raw_gradients": raw_gradients,
            "clipped_gradients": clipped_gradients,
            "optimizer_before": before_optimizer,
            "optimizer_after": after_optimizer,
            "parameter_state_before": before_parameters,
            "parameter_state_after": after_parameters,
            "batch_size": result["batch_size"],
            "row_ids": result["row_ids"],
            "optimizer_step": result["optimizer_step"],
            "parameter_state_sha256": result["parameter_state_sha256"],
        }

    def close(self) -> None:
        self.hook.remove()


def production_reference(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed504_manifest = json.loads(
        (ROOT / "docs/data/seed504/manifest.json").read_text()
    )
    train_sources = seed504_manifest["sources"]["training"]
    identities = [
        f"{source['path']}:{line}"
        for source in train_sources
        for line, _row in enumerate(
            (ROOT / source["path"]).read_bytes().splitlines(), 1
        )
    ]
    paths = [ROOT / source["path"] for source in train_sources]
    x, p, v, _replay, weights = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    if len(identities) != len(x) or len(x) != 13724:
        raise ValueError("v2_production_reference_loader_identity_mismatch")
    recorder = ProductionRecorder(identities, (x, p, v, weights))
    original_epoch = train.train_one_epoch
    original_clip = torch.nn.utils.clip_grad_norm_

    def instrumented_epoch(**kwargs: Any) -> dict[str, Any]:
        recorder.attach(kwargs["model"])
        kwargs["step_callback"] = recorder.step_callback
        kwargs["step_callback_needs_raw_gradients"] = True
        kwargs["loss_observer"] = recorder.loss_observer
        return original_epoch(**kwargs)

    def instrumented_clip(
        parameters: Any, max_norm: float, *args: Any, **kwargs: Any
    ) -> Any:
        result = original_clip(parameters, max_norm, *args, **kwargs)
        recorder.latest_clip_return = float(result.detach().cpu().item())
        return result

    train.train_one_epoch = instrumented_epoch
    torch.nn.utils.clip_grad_norm_ = instrumented_clip
    previous_initializer = run_seed504.INITIALIZER
    run_seed504.INITIALIZER = CORPUS / "initializer.npz"
    try:
        run_seed504.run(out, do_train=True)
    finally:
        train.train_one_epoch = original_epoch
        torch.nn.utils.clip_grad_norm_ = original_clip
        run_seed504.INITIALIZER = previous_initializer
        recorder.close()
    if len(recorder.records) != 108:
        raise ValueError(
            f"v2_production_reference_update_count:{len(recorder.records)}"
        )
    trace_path = out / "production-updates.jsonl"
    trace_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in recorder.records
        )
    )
    return recorder.records, {
        "invocation": "unmodified ml.alphazero_lite.run_seed504.run(out=production_reference_dir, do_train=True); initializer path relocated to authenticated seed542 initializer",
        "trace_sha256": sha(trace_path),
        "production_runner_sha256": sha(Path(run_seed504.__file__)),
        "updates": 108,
    }


def compare_records(production: dict[str, Any], direct: dict[str, Any]) -> list[str]:
    fields = (
        "update",
        "batch",
        "forward",
        "policy_loss",
        "value_loss",
        "weighted_total_objective",
        "gradient_norm_preclip",
        "gradient_norm_clip_function",
        "raw_gradients",
        "clipped_gradients",
        "optimizer_before",
        "optimizer_after",
        "parameter_state_before",
        "parameter_state_after",
    )
    return [
        field
        for field in fields
        if canonical(production.get(field)) != canonical(direct.get(field))
    ]


def run_direct_arm(
    arm: str,
    out: Path,
    schedule: list[dict[str, Any]],
    reference: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    x, p, v, weights, identities = fixed_update.load_sources(arm)
    positions = {identity: index for index, identity in enumerate(identities)}
    model = fixed_update.load_model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    trace_path = out / f"{arm}-updates.jsonl"
    checkpoints = {}
    updates_completed = 0
    validity = []
    boundary_checks = []
    recorder = DirectRecorder(model, x, p, v, weights, identities)
    with trace_path.open("w", encoding="utf-8") as stream:
        for item in schedule:
            record = recorder.update(optimizer, item[arm], positions)
            record["update"] = item["update"]
            if arm == "C" and reference is not None:
                mismatches = compare_records(reference[item["update"] - 1], record)
                validity.append(
                    {
                        "update": item["update"],
                        "bitwise_equal": not mismatches,
                        "mismatched_fields": mismatches,
                    }
                )
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            updates_completed += 1
            if item["update"] in BOUNDARIES:
                checkpoint = out / f"{arm}-update-{item['update']:03d}.npz"
                checkpoint_arrays = train.checkpoint_from_model(model)
                np.savez(checkpoint, **checkpoint_arrays)
                checkpoints[str(item["update"])] = sha(checkpoint)
                if arm == "C":
                    reference_checkpoint = (
                        out
                        / "production-reference"
                        / f"seed504_epoch{item['update'] // 27}.npz"
                    )
                    with np.load(reference_checkpoint) as reference_arrays:
                        matches = set(reference_arrays.files) == set(
                            checkpoint_arrays
                        ) and all(
                            checkpoint_arrays[name].shape
                            == reference_arrays[name].shape
                            and checkpoint_arrays[name].dtype
                            == reference_arrays[name].dtype
                            and np.array_equal(
                                checkpoint_arrays[name], reference_arrays[name]
                            )
                            for name in checkpoint_arrays
                        )
                    boundary_checks.append(
                        {
                            "update": item["update"],
                            "production_checkpoint_bitwise_equal": matches,
                        }
                    )
                    if not matches:
                        break
            if arm == "C" and validity[-1]["bitwise_equal"] is False:
                break
    summary = {
        "arm": arm,
        "status": "current_runtime_control_passed"
        if arm == "C"
        and updates_completed == 108
        and len(boundary_checks) == 4
        and all(row["production_checkpoint_bitwise_equal"] for row in boundary_checks)
        else "training_completed"
        if arm == "D" and updates_completed == 108
        else "current_runtime_control_failed",
        "updates_completed": updates_completed,
        "exposures_completed": sum(
            record["batch_size"] for record in _records(trace_path)
        ),
        "trace_sha256": sha(trace_path),
        "checkpoints": checkpoints,
        "production_vs_C_update_checks": validity if arm == "C" else None,
        "boundary_production_checkpoint_checks": boundary_checks
        if arm == "C"
        else None,
    }
    (out / f"{arm}-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if arm == "C":
        (out / "current-runtime-control-validity.json").write_text(
            json.dumps(
                {
                    "passed": summary["status"] == "current_runtime_control_passed"
                    and all(item["bitwise_equal"] for item in validity),
                    "production_trace_sha256": sha(
                        out / "production-reference/production-updates.jsonl"
                    ),
                    "C_trace_sha256": sha(trace_path),
                    "updates": validity,
                    "boundary_checkpoints": boundary_checks,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return summary


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_prior_disposition(out: Path) -> None:
    disposition = {
        "prior_attempts_preserved": [
            "docs/data/seed542-fixed-update-cd/",
            "docs/data/seed542-fixed-update-cd-reproduction/",
            "docs/data/seed542-fixed-update-cd-diagnostics/",
        ],
        "original_attempt": "had a zero-based/one-based update-indexing error and stopped at an incorrectly labeled boundary",
        "corrected_runner_diagnostic": "matched production training across 108 updates; the first diagnostic's only unequal field was a diagnostic gradient-norm calculation discrepancy, not a parameter/update discrepancy",
        "historical_seed504": "current-runtime production did not reproduce historical seed504 E1 bitwise; exact cause remains unknown",
        "prior_D_execution": False,
        "causal_attribution": "none; no attribution to PyTorch version, threading, or hardware",
        "new_protocol": "current-runtime production-versus-C bitwise equality is the validity gate; historical checkpoint comparisons are descriptive",
    }
    (out / "prior-diagnostic-disposition.json").write_text(
        json.dumps(disposition, indent=2, sort_keys=True) + "\n"
    )


def write_readme(out: Path, status: str, *, d_executed: bool) -> None:
    text = f"""# Seed542 current-runtime fixed-update C/D comparison v2

Status: **{status}**. This is a new prospective current-runtime production-control experiment, not an exact historical seed504 reproduction.

The original attempt had an update-indexing error. The corrected runner matched production training across 108 diagnostic updates. Current-runtime production did not reproduce historical seed504 E1 bitwise; the cause remains unknown. No attribution is made to PyTorch, threading, or hardware. D was not executed in the prior attempts.

This registration consumes the unchanged seed542 published per-arm one-based schedules and checks each against the merged zero-based schedule view before execution. The training contract is fixed at 108 updates and 54,896 exposures per arm, batches `[512] * 26 + [412]` repeated four times, residual_v3/kalah_v3 (96, 3), legal policy CE + 0.5 × Huber(value, delta=1), fresh Adam (lr 0.0001, betas (0.9,0.999), eps 1e-8), no weight decay or scheduler, and gradient clipping 1.0.

The unmodified seed504 production runner is used as the current-runtime reference. Its initializer path is relocated at runtime to the authenticated, byte-identical seed542 initializer. Non-mutating callbacks record the production update and current direct-C update for exact all-step comparison. Historical seed504 checkpoint differences are descriptive only. D execution status: `{d_executed}`.

Registration and evidence are in this directory. `execution-binding.json` records numerical settings and resource limits. `production-reference/` contains the production runner outputs and update trace; `C-updates.jsonl` and `D-updates.jsonl` contain row identities and update telemetry; boundary checkpoints are retained at 27, 54, 81, and 108. `development-predictions.jsonl` and `development-metrics.json` cover only the unchanged retained membership and are descriptive; negative D-minus-control loss differences indicate improvement. No development threshold is used.

The intended later composition remains D-final policy/trunk with the frozen initializer value/trunk. Trained value metrics are diagnostics only. No arena, promotion, serving export, or deployment is part of this experiment.
"""
    (out / "README.md").write_text(text)


def historical_drift(out: Path) -> dict[str, Any]:
    comparisons = {}
    for epoch in range(1, 5):
        current_path = out / "production-reference" / f"seed504_epoch{epoch}.npz"
        old_path = ROOT / f"docs/data/seed504/seed504_epoch{epoch}.npz"
        with np.load(current_path) as current, np.load(old_path) as historical:
            differences = {}
            total_mismatch = 0
            for name in sorted(set(current.files) | set(historical.files)):
                if name not in current or name not in historical:
                    differences[name] = {"parameter_presence_mismatch": True}
                    continue
                left, right = current[name], historical[name]
                count = int(np.count_nonzero(left != right))
                total_mismatch += count
                differences[name] = {
                    "shape_equal": left.shape == right.shape,
                    "dtype_equal": left.dtype == right.dtype,
                    "bitwise_equal": count == 0
                    and left.shape == right.shape
                    and left.dtype == right.dtype,
                    "differing_elements": count,
                    "maximum_absolute_difference": float(
                        np.max(
                            np.abs(left.astype(np.float64) - right.astype(np.float64)),
                            initial=0.0,
                        )
                    ),
                }
            comparisons[f"E{epoch}"] = {
                "historical_checkpoint_sha256": sha(old_path),
                "current_runtime_production_checkpoint_sha256": sha(current_path),
                "different_elements": total_mismatch,
                "parameter_names_equal": set(current.files) == set(historical.files),
                "bitwise_equal": total_mismatch == 0
                and set(current.files) == set(historical.files)
                and all(
                    row.get("bitwise_equal", False) for row in differences.values()
                ),
                "comparison_is_descriptive_only": True,
                "parameters": differences,
            }
    (out / "historical-seed504-drift.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n"
    )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tablebase", type=Path, default=TABLEBASE)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if tuple(os.environ.get(key) for key in ENVIRONMENT_KEYS) != ("1",) * len(
        ENVIRONMENT_KEYS
    ):
        raise ValueError("seed542_v2_thread_environment_must_be_explicitly_one")
    # Full read-only authentication occurs before the prospective registration.
    verify_seed542_amended_corpus.verify(root=ROOT, tablebase_path=args.tablebase)
    schedule_doc = fixed_update.load_schedule()
    commands = [
        f"production reference and C/D runner: {os.environ.get('SEED542_CD_LAUNCH_COMMAND')}",
        f"independent verifier: {sys.executable} -m ml.alphazero_lite.verify_seed542_fixed_update_cd_v2 --root <physical-copy> --out <physical-copy>/{args.out.relative_to(ROOT)} --tablebase {args.tablebase}",
    ]
    if args.prepare_only:
        if args.out.exists() and any(args.out.iterdir()):
            raise FileExistsError(f"seed542_v2_preflight_output_not_empty:{args.out}")
        args.out.mkdir(parents=True, exist_ok=True)
        registration = freeze_registration(args.out, commands, args.tablebase)
        write_prior_disposition(args.out)
        (args.out / "preflight-status.json").write_text(
            json.dumps(
                {
                    "status": "prospective sources and inputs frozen; no reference or training executed",
                    "registration_sha256": sha(args.out / "registration.json"),
                    "prepare_command": sys.argv,
                    "planned_execution_command": os.environ.get(
                        "SEED542_CD_LAUNCH_COMMAND"
                    ),
                    "schedule_updates": len(schedule_doc["updates"]),
                    "initializer_sha256": sha(CORPUS / "initializer.npz"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return
    if not args.out.is_dir() or not (args.out / "preflight-status.json").is_file():
        raise FileNotFoundError("seed542_v2_prospective_preflight_not_found")
    if (args.out / "execution-binding.json").exists():
        raise FileExistsError(
            "seed542_v2_execution_already_started; preserve existing attempt"
        )
    registration = verify_frozen_registration(args.out)
    preflight = json.loads((args.out / "preflight-status.json").read_text())
    if preflight["registration_sha256"] != sha(args.out / "registration.json"):
        raise ValueError("seed542_v2_preflight_registration_hash_mismatch")
    if (
        preflight["status"]
        != "prospective sources and inputs frozen; no reference or training executed"
    ):
        raise ValueError("seed542_v2_preflight_status_invalid")
    (args.out / "execution-binding.json").write_text(
        json.dumps(
            {
                "registration_sha256": sha(args.out / "registration.json"),
                "registration_source_hashes": registration["execution_source_sha256"],
                "runtime_and_resource_settings": environment_settings(),
                "actual_command": sys.argv,
                "outer_command": os.environ.get("SEED542_CD_LAUNCH_COMMAND"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    production_dir = args.out / "production-reference"
    production_dir.mkdir()
    # The production runner's implementation is unmodified; only its initializer
    # path is relocated to the authenticated, byte-identical seed542 asset.
    production_records, production_summary = production_reference(production_dir)
    (args.out / "production-reference-summary.json").write_text(
        json.dumps(production_summary, indent=2, sort_keys=True) + "\n"
    )
    historical_drift(args.out)
    control = run_direct_arm("C", args.out, schedule_doc["updates"], production_records)
    control_valid = (
        control["updates_completed"] == 108
        and all(
            row["bitwise_equal"] for row in control["production_vs_C_update_checks"]
        )
        and len(control["boundary_production_checkpoint_checks"]) == 4
        and all(
            row["production_checkpoint_bitwise_equal"]
            for row in control["boundary_production_checkpoint_checks"]
        )
    )
    if not control_valid:
        first_update_mismatch = next(
            (
                row
                for row in control["production_vs_C_update_checks"]
                if not row["bitwise_equal"]
            ),
            None,
        )
        (args.out / "run-status.json").write_text(
            json.dumps(
                {
                    "status": "current_runtime_control_failed",
                    "D_executed": False,
                    "C_updates_completed": control["updates_completed"],
                    "first_mismatch": first_update_mismatch,
                    "boundary_mismatches": [
                        row
                        for row in control["boundary_production_checkpoint_checks"]
                        if not row["production_checkpoint_bitwise_equal"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        write_readme(
            args.out,
            "current-runtime control failed; D not executed",
            d_executed=False,
        )
        return
    expanded = run_direct_arm("D", args.out, schedule_doc["updates"])
    if expanded["updates_completed"] != 108:
        raise RuntimeError("seed542_v2_D_update_count_incomplete")
    evaluation = publication.publish(args.out)
    (args.out / "run-status.json").write_text(
        json.dumps(
            {
                "status": "training_and_development_evaluation_completed",
                "D_executed": True,
                "C_control_passed": True,
                "development_effects": evaluation["effects"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_readme(
        args.out,
        "current-runtime training and development evaluation completed",
        d_executed=True,
    )


if __name__ == "__main__":
    main()
