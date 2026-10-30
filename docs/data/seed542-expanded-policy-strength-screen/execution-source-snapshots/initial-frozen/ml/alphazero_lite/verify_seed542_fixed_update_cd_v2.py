"""Independent read-only verification of seed542 current-runtime C/D v2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train, verify_seed542_amended_corpus

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
BOUNDARIES = {27, 54, 81, 108}
BUCKETS = (">32", "17-32", "<=16")


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


def model_state_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b"\0")
        h.update(str(array.dtype).encode() + b"\0")
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


def named_parameter_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b"\0")
        h.update(str(array.dtype).encode() + b"\0")
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


def optimizer_snapshot(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> dict[str, Any]:
    parameters = list(model.named_parameters())
    state = {}
    for name, parameter in parameters:
        values = optimizer.state.get(parameter, {})
        state[name] = {
            key: array_record(value) if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }
    return {
        "parameter_order": [name for name, _parameter in parameters],
        "groups": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
        "state": state,
    }


def published_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    merged = protocol["updates"]
    c_schedule = protocol["control"]["schedule"]
    d_schedule = protocol["expanded"]["schedule"]
    if not (len(merged) == len(c_schedule) == len(d_schedule) == 108):
        raise ValueError("v2_published_schedule_update_count_mismatch")
    expected_sizes = [size for _ in range(4) for size in ([512] * 26 + [412])]
    result = []
    for update, (view, c_row, d_row, size) in enumerate(
        zip(merged, c_schedule, d_schedule, expected_sizes, strict=True), 1
    ):
        if (
            view["update"] != update - 1
            or c_row["update"] != update
            or d_row["update"] != update
            or view["batch_size"] != size
            or c_row["batch_size"] != size
            or d_row["batch_size"] != size
            or view["C"] != c_row["rows"]
            or view["D"] != d_row["rows"]
        ):
            raise ValueError(f"v2_published_schedule_view_mismatch:{update}")
        result.append(
            {
                "update": update,
                "batch_size": size,
                "C": c_row["rows"],
                "D": d_row["rows"],
            }
        )
    c_rows = [row for item in result for row in item["C"]]
    d_rows = [row for item in result for row in item["D"]]
    if len(c_rows) != 54896 or len(d_rows) != 54896 or len(set(d_rows)) != 54896:
        raise ValueError("v2_published_schedule_exposure_mismatch")
    return result


def validate_record_pair(
    production: dict[str, Any], direct: dict[str, Any]
) -> list[str]:
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


def compare_checkpoint_arrays(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> bool:
    return set(left) == set(right) and all(
        left[name].shape == right[name].shape
        and left[name].dtype == right[name].dtype
        and np.array_equal(left[name], right[name])
        for name in left
    )


def validate_control_gate(validity: dict[str, Any]) -> None:
    updates = validity.get("updates", [])
    boundaries = validity.get("boundary_checkpoints", [])
    if (
        validity.get("passed") is not True
        or len(updates) != 108
        or [row.get("update") for row in updates] != list(range(1, 109))
        or not all(row.get("bitwise_equal") is True for row in updates)
        or len(boundaries) != 4
        or [row.get("update") for row in boundaries] != [27, 54, 81, 108]
        or not all(
            row.get("production_checkpoint_bitwise_equal") is True for row in boundaries
        )
    ):
        raise ValueError("v2_current_runtime_control_gate_failed")


def assert_json_equal(expected: Any, actual: Any, code: str) -> None:
    if canonical(expected) != canonical(actual):
        raise ValueError(code)


def authenticate(
    root: Path, output: Path, tablebase: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    amendment = json.loads(
        (
            root / "docs/data/seed542-selfplay-4096/continuation-amendment.json"
        ).read_text()
    )
    registration_path = output / "registration.json"
    binding_path = output / "execution-binding.json"
    registration = json.loads(registration_path.read_text())
    binding = json.loads(binding_path.read_text())
    asset_binding = registration["runtime_assets"]
    if (
        sha(tablebase) != asset_binding["tablebase_sha256"]
        or tablebase.stat().st_size != asset_binding["tablebase_size_bytes"]
    ):
        raise ValueError("v2_tablebase_runtime_asset_binding_mismatch")
    verify_seed542_amended_corpus.verify(root=root, tablebase_path=tablebase)
    registered_runtime = dict(registration["runtime_and_resource_settings"])
    execution_runtime = dict(binding["runtime_and_resource_settings"])
    registered_runtime.pop("cgroup", None)
    execution_runtime.pop("cgroup", None)
    if canonical(registered_runtime) != canonical(execution_runtime):
        raise ValueError("v2_registration_execution_runtime_mismatch")
    if binding["registration_sha256"] != sha(registration_path):
        raise ValueError("v2_registration_sha256_mismatch")
    if binding["registration_source_hashes"] != registration["execution_source_sha256"]:
        raise ValueError("v2_execution_binding_source_set_mismatch")
    for relative, expected in registration["input_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"v2_input_sha256_mismatch:{relative}")
    for relative, expected in registration["execution_source_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"v2_execution_source_sha256_mismatch:{relative}")
    current = binding["runtime_and_resource_settings"]
    if current["python"] != sys.version or current["platform"] != platform.platform():
        raise ValueError("v2_verifier_runtime_mismatch")
    if current["numpy"] != np.__version__ or current["torch"] != torch.__version__:
        raise ValueError("v2_verifier_numeric_package_mismatch")
    packages = sorted(
        (
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
            and distribution.metadata["Name"].lower() != "pip"
        ),
        key=str.casefold,
    )
    if packages != amendment["interpreter"]["installed_packages"]:
        raise ValueError("v2_verifier_complete_package_set_mismatch")
    if (
        current["device"] != "cpu"
        or current["dtype"] != "torch.float32 / numpy.float32"
    ):
        raise ValueError("v2_numeric_execution_contract_mismatch")
    if (
        current["torch_deterministic_algorithms"]
        != torch.are_deterministic_algorithms_enabled()
        or current["mkldnn_enabled"] != torch.backends.mkldnn.enabled
        or current["cudnn_enabled"] != torch.backends.cudnn.enabled
        or current["cudnn_allow_tf32"] != torch.backends.cudnn.allow_tf32
        or current["cuda_matmul_allow_tf32"] != torch.backends.cuda.matmul.allow_tf32
        or current["float32_matmul_precision"] != torch.get_float32_matmul_precision()
        or current["torch_build_config"] != torch.__config__.show()
        or current["cuda_available"] != torch.cuda.is_available()
    ):
        raise ValueError("v2_backend_setting_mismatch")
    if current["torch_num_threads"] != 1 or current["torch_num_interop_threads"] != 1:
        raise ValueError("v2_thread_count_contract_mismatch")
    if current["environment_variables"] != {key: "1" for key in ENVIRONMENT_KEYS}:
        raise ValueError("v2_thread_environment_contract_mismatch")
    expected_resources = current["resource_limits"]
    if (
        expected_resources["requested_cpu_quota"] != "25%"
        or expected_resources["requested_cpu_weight"] != "1"
        or expected_resources["requested_memory_max"] != "4294967296"
        or expected_resources["requested_allowed_cpus"] != "30"
        or expected_resources["requested_nice"] != "19"
        or expected_resources["requested_address_space_limit"] != "17179869184"
    ):
        raise ValueError("v2_requested_resource_contract_mismatch")
    actual_cgroup = _cgroup_settings()
    if (
        sorted(os.sched_getaffinity(0)) != current["cpu_affinity"]
        or os.nice(0) != 19
        or list(resource.getrlimit(resource.RLIMIT_AS))
        != expected_resources["address_space_rlimit"]
        or actual_cgroup
        != {
            "cpu_max": expected_resources["cpu_max"],
            "cpu_weight": expected_resources["cpu_weight"],
            "memory_max": expected_resources["memory_max"],
            "cpuset_effective": expected_resources["cpuset_effective"],
        }
    ):
        raise ValueError("v2_verifier_resource_limit_mismatch")
    protocol = json.loads(
        (root / "docs/data/seed542-selfplay-4096/training-protocol.json").read_text()
    )
    schedule = published_schedule(protocol)
    if registration["protocol_sha256"] != sha(
        root / "docs/data/seed542-selfplay-4096/training-protocol.json"
    ):
        raise ValueError("v2_protocol_sha256_mismatch")
    return registration, schedule


def _cgroup_settings() -> dict[str, str | None]:
    members = Path("/proc/self/cgroup").read_text().splitlines()
    v2_path = next(row.split("::", 1)[1] for row in members if row.startswith("0::"))
    folder = Path("/sys/fs/cgroup") / v2_path.lstrip("/")
    result = {}
    for name in ("cpu.max", "cpu.weight", "memory.max", "cpuset.cpus.effective"):
        try:
            result[name] = (folder / name).read_text().strip()
        except OSError:
            result[name] = None
    return {
        "cpu_max": result["cpu.max"],
        "cpu_weight": result["cpu.weight"],
        "memory_max": result["memory.max"],
        "cpuset_effective": result["cpuset.cpus.effective"],
    }


def load_arm_data(
    arm: str, root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if arm == "C":
        manifest = json.loads((root / "docs/data/seed504/manifest.json").read_text())
        sources = manifest["sources"]["training"]
    else:
        manifest = json.loads(
            (root / "docs/data/seed542-selfplay-4096/source-manifest.json").read_text()
        )
        sources = [
            source for source in manifest["sources"] if source["role"] == "training"
        ]
    paths = [root / source["path"] for source in sources]
    row_ids = []
    for source, path in zip(sources, paths, strict=True):
        if sha(path) != source["sha256"]:
            raise ValueError(f"v2_training_source_mismatch:{source['path']}")
        row_ids.extend(
            f"{source['path']}:{line}"
            for line, _record in enumerate(path.read_bytes().splitlines(), 1)
        )
    x, p, v, _replay, weights = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    if len(row_ids) != len(x):
        raise ValueError(f"v2_loader_row_count_mismatch:{arm}")
    return x, p, v, weights, row_ids


def replay_arm(
    arm: str,
    root: Path,
    output: Path,
    schedule: list[dict[str, Any]],
    production_records: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    x, p, v, policy_weights, identities = load_arm_data(arm, root)
    positions = {identity: position for position, identity in enumerate(identities)}
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        model, root / "docs/data/seed542-selfplay-4096/initializer.npz"
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    direct_records = [
        json.loads(line)
        for line in (output / f"{arm}-updates.jsonl").read_text().splitlines()
    ]
    if len(direct_records) != 108:
        raise ValueError(f"v2_trace_update_count_mismatch:{arm}")
    for item, published in zip(schedule, direct_records, strict=True):
        rows = item[arm]
        update_index = item["update"]
        if (
            published["update"] != update_index
            or published["row_ids"] != rows
            or published["batch_size"] != item["batch_size"]
        ):
            raise ValueError(
                f"v2_trace_schedule_identity_mismatch:{arm}:{update_index}"
            )
        indexes = np.asarray([positions[row_id] for row_id in rows], dtype=np.int64)
        x_batch = torch.from_numpy(x[indexes])
        p_batch = torch.from_numpy(p[indexes])
        v_batch = torch.from_numpy(v[indexes])
        w_batch = torch.from_numpy(policy_weights[indexes])
        legal_array = train.legal_mask_matrix_for_encoded_states(x[indexes])
        legal = torch.from_numpy(legal_array)
        model.train()
        before_params = model_state_digest(model)
        before_optimizer = optimizer_snapshot(optimizer, model)
        logits, value_prediction = model(x_batch)
        forward = {
            "input": array_record(x_batch),
            "policy_logits": array_record(logits),
            "value_prediction": array_record(value_prediction),
            "model_training": bool(model.training),
        }
        objective_policy = train.weighted_policy_loss(
            train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0, -1e9), p_batch
            ),
            w_batch,
        )
        objective_value = train.compute_value_loss_vector(
            value_prediction, v_batch, value_loss="huber", huber_delta=1.0
        ).mean()
        objective = objective_policy + 0.5 * objective_value
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        raw_gradients = {
            name: array_record(
                parameter.grad
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
            for name, parameter in model.named_parameters()
        }
        production_gradient_squared = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                production_gradient_squared += float(
                    torch.sum(parameter.grad.detach() ** 2).item()
                )
        production_logged_gradient_norm = float(np.sqrt(production_gradient_squared))
        clip_return = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped_gradients = {
            name: array_record(
                parameter.grad
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
            for name, parameter in model.named_parameters()
        }
        clip_norm = float(clip_return.detach().cpu().item())
        optimizer.step()
        after_optimizer = optimizer_snapshot(optimizer, model)
        after_params = model_state_digest(model)
        batch = {
            "row_ids": rows,
            "batch_size": len(rows),
            "tensors": {
                "input": array_record(x_batch),
                "target_policy": array_record(p_batch),
                "target_value": array_record(v_batch),
                "policy_weight": array_record(w_batch),
                "legal_mask": array_record(legal),
            },
        }
        replayed = {
            "update": update_index,
            "batch": batch,
            "forward": forward,
            "policy_loss": float(objective_policy.detach().cpu().item()),
            "value_loss": float(objective_value.detach().cpu().item()),
            "weighted_total_objective": float(objective.detach().cpu().item()),
            "gradient_norm_preclip": clip_norm,
            "gradient_norm_clip_function": clip_norm,
            "raw_gradients": raw_gradients,
            "clipped_gradients": clipped_gradients,
            "optimizer_before": before_optimizer,
            "optimizer_after": after_optimizer,
            "parameter_state_before": before_params,
            "parameter_state_after": after_params,
            "batch_size": len(rows),
            "row_ids": rows,
            "optimizer_step": int(
                optimizer.state[next(model.parameters())]["step"].item()
            ),
            "parameter_state_sha256": _parameter_digest(model),
        }
        assert_json_equal(
            replayed, published, f"v2_independent_replay_mismatch:{arm}:{update_index}"
        )
        if arm == "C":
            if production_records is None:
                raise ValueError("v2_production_trace_missing")
            if (
                production_records[update_index - 1].get(
                    "production_logged_gradient_norm"
                )
                != production_logged_gradient_norm
            ):
                raise ValueError(
                    f"v2_production_gradient_telemetry_mismatch:{update_index}"
                )
            mismatches = validate_record_pair(
                production_records[update_index - 1], replayed
            )
            if mismatches:
                raise ValueError(
                    f"v2_production_C_mismatch:{update_index}:{','.join(mismatches)}"
                )
        if update_index in BOUNDARIES:
            checkpoint_path = output / f"{arm}-update-{update_index:03d}.npz"
            with np.load(checkpoint_path) as saved:
                current = train.checkpoint_from_model(model)
                expected = {name: saved[name] for name in saved.files}
                if not compare_checkpoint_arrays(current, expected):
                    raise ValueError(
                        f"v2_checkpoint_parameter_mismatch:{arm}:{update_index}"
                    )
    return direct_records


def _parameter_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b"\0")
        h.update(str(value.dtype).encode() + b"\0")
        h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        h.update(value.tobytes())
    return h.hexdigest()


def _read_dev(root: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads(
        (root / "docs/data/seed542-selfplay-4096/source-manifest.json").read_text()
    )
    sources = [
        source for source in manifest["sources"] if source["role"] == "development"
    ]
    membership = json.loads(
        (
            root / "docs/data/seed542-selfplay-4096/development-membership.json"
        ).read_text()
    )
    included = {row["row_id"] for row in membership["membership"] if row["included"]}
    paths = [root / source["path"] for source in sources]
    identities = []
    for source, path in zip(sources, paths, strict=True):
        if sha(path) != source["sha256"]:
            raise ValueError(f"v2_dev_source_hash_mismatch:{source['path']}")
        identities.extend(
            f"{source['path']}:{line}"
            for line, _ in enumerate(path.read_bytes().splitlines(), 1)
        )
    x, p, v, _replay, _weights = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    select = [index for index, row_id in enumerate(identities) if row_id in included]
    if len(select) != len(included):
        raise ValueError("v2_dev_membership_count_mismatch")
    return [identities[index] for index in select], x[select], p[select], v[select, 0]


def independently_recompute_development(
    root: Path, output: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names, x, p, values = _read_dev(root)
    endpoint_paths = {
        "initializer": root / "docs/data/seed542-selfplay-4096/initializer.npz",
        "C-final": output / "C-update-108.npz",
        "D-final": output / "D-update-108.npz",
    }
    metric_rows = {}
    prediction_rows = []
    for endpoint, checkpoint in endpoint_paths.items():
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(model, checkpoint)
        model.eval()
        legal = train.legal_mask_matrix_for_encoded_states(x).astype(bool)
        with torch.no_grad():
            logits_tensor, value_tensor = model(torch.from_numpy(x))
        logits = logits_tensor.numpy().copy()
        logits[~legal] = -1e9
        log_probs = logits - np.logaddexp.reduce(logits, axis=1)[:, None]
        probabilities = np.exp(log_probs)
        predictions = value_tensor.numpy().reshape(-1)
        ce = -(p * log_probs).sum(axis=1)
        mse = (predictions - values) ** 2
        groups: dict[str, list[int]] = {
            "pooled": list(range(len(names))),
            **{name: [] for name in BUCKETS},
        }
        for index in range(len(names)):
            stones = int(round(float(np.sum(x[index, :12])) * 48.0))
            group = ">32" if stones > 32 else "17-32" if stones >= 17 else "<=16"
            groups[group].append(index)
        cohort_metrics = {}
        for group, indices in groups.items():
            ids: dict[bytes, list[int]] = defaultdict(list)
            for index in indices:
                ids[x[index].tobytes()].append(index)
            cohort_metrics[group] = {
                "rows": len(indices),
                "production_float32_input_identities": len(ids),
                "exposure_weighted": {
                    "policy_ce": float(ce[indices].mean()),
                    "actual_value_mse": float(mse[indices].mean()),
                }
                if indices
                else None,
                "equal_production_float32_input_weighted": {
                    "policy_ce": float(np.mean([ce[ix].mean() for ix in ids.values()])),
                    "actual_value_mse": float(
                        np.mean([mse[ix].mean() for ix in ids.values()])
                    ),
                }
                if indices
                else None,
            }
        metric_rows[endpoint] = {
            "checkpoint_sha256": sha(checkpoint),
            "cohort_row_count": len(names),
            "cohort_identity_count": len({state.tobytes() for state in x}),
            "metrics": cohort_metrics,
        }
        prediction_rows.extend(
            {
                "endpoint": endpoint,
                "row_id": names[index],
                "production_float32_input_sha256": hashlib.sha256(
                    x[index].tobytes()
                ).hexdigest(),
                "target_policy": p[index].tolist(),
                "target_actual_value": float(values[index]),
                "legal_mask": legal[index].tolist(),
                "policy_logits_legal_masked": logits[index].tolist(),
                "policy_probabilities_legal_masked": probabilities[index].tolist(),
                "value_prediction": float(predictions[index]),
            }
            for index in range(len(names))
        )
    effects = {}
    for effect_name, new, baseline in (
        ("D_minus_C", "D-final", "C-final"),
        ("D_minus_initializer", "D-final", "initializer"),
    ):
        effects[effect_name] = {}
        for group in ("pooled", *BUCKETS):
            effects[effect_name][group] = {}
            for weighting in (
                "exposure_weighted",
                "equal_production_float32_input_weighted",
            ):
                new_values = metric_rows[new]["metrics"][group][weighting]
                base_values = metric_rows[baseline]["metrics"][group][weighting]
                effects[effect_name][group][weighting] = {
                    metric: new_values[metric] - base_values[metric]
                    if new_values is not None and base_values is not None
                    else None
                    for metric in ("policy_ce", "actual_value_mse")
                }
    report = {
        "development_membership_sha256": sha(
            root / "docs/data/seed542-selfplay-4096/development-membership.json"
        ),
        "sign_convention": "effect = first endpoint loss minus second endpoint loss; negative means improvement",
        "cohort_counts": {
            group: {
                "rows": values["rows"],
                "production_float32_input_identities": values[
                    "production_float32_input_identities"
                ],
            }
            for group, values in metric_rows["initializer"]["metrics"].items()
        },
        "metrics": {
            endpoint: values["metrics"] for endpoint, values in metric_rows.items()
        },
        "effects": effects,
        "endpoints": {
            endpoint: {
                key: values[key]
                for key in (
                    "checkpoint_sha256",
                    "cohort_row_count",
                    "cohort_identity_count",
                )
            }
            for endpoint, values in metric_rows.items()
        },
        "interpretation": "Descriptive diagnostics only; no advancement threshold. Intended later runtime composition is D-final policy/trunk plus frozen initializer value/trunk; trained value metrics are diagnostic only.",
    }
    return report, prediction_rows


def verify_historical_drift(root: Path, output: Path) -> None:
    published = json.loads((output / "historical-seed504-drift.json").read_text())
    actual = {}
    for epoch in range(1, 5):
        reference = output / "production-reference" / f"seed504_epoch{epoch}.npz"
        historical = root / f"docs/data/seed504/seed504_epoch{epoch}.npz"
        with np.load(reference) as current, np.load(historical) as old:
            params = {}
            mismatches = 0
            for name in sorted(set(current.files) | set(old.files)):
                if name not in current or name not in old:
                    params[name] = {"parameter_presence_mismatch": True}
                    continue
                a, b = current[name], old[name]
                count = int(np.count_nonzero(a != b))
                mismatches += count
                params[name] = {
                    "shape_equal": a.shape == b.shape,
                    "dtype_equal": a.dtype == b.dtype,
                    "bitwise_equal": count == 0
                    and a.shape == b.shape
                    and a.dtype == b.dtype,
                    "differing_elements": count,
                    "maximum_absolute_difference": float(
                        np.max(
                            np.abs(a.astype(np.float64) - b.astype(np.float64)),
                            initial=0.0,
                        )
                    ),
                }
        actual[f"E{epoch}"] = {
            "historical_checkpoint_sha256": sha(historical),
            "current_runtime_production_checkpoint_sha256": sha(reference),
            "different_elements": mismatches,
            "parameter_names_equal": set(current.files) == set(old.files),
            "bitwise_equal": mismatches == 0
            and set(current.files) == set(old.files)
            and all(row.get("bitwise_equal", False) for row in params.values()),
            "comparison_is_descriptive_only": True,
            "parameters": params,
        }
    assert_json_equal(actual, published, "v2_historical_drift_report_mismatch")


def verify(root: Path, output: Path, tablebase: Path) -> dict[str, Any]:
    registration = json.loads((output / "registration.json").read_text())
    observed_paths = {
        *(root / relative for relative in registration["input_sha256"]),
        tablebase,
        *(path for path in output.rglob("*") if path.is_file()),
    }
    before_snapshot = _file_snapshot(observed_paths)
    registration, schedule = authenticate(root, output, tablebase)
    status = json.loads((output / "run-status.json").read_text())
    if (
        status["status"] != "training_and_development_evaluation_completed"
        or not status["C_control_passed"]
        or not status["D_executed"]
    ):
        raise ValueError("v2_experiment_not_complete")
    production = [
        json.loads(line)
        for line in (output / "production-reference/production-updates.jsonl")
        .read_text()
        .splitlines()
    ]
    if len(production) != 108:
        raise ValueError("v2_production_reference_trace_count_mismatch")
    replay_arm("C", root, output, schedule, production)
    validity = json.loads(
        (output / "current-runtime-control-validity.json").read_text()
    )
    validate_control_gate(validity)
    replay_arm("D", root, output, schedule, None)
    verify_historical_drift(root, output)
    dev_metrics, dev_predictions = independently_recompute_development(root, output)
    actual_metrics = json.loads((output / "development-metrics.json").read_text())
    assert_json_equal(
        dev_metrics, actual_metrics, "v2_development_metrics_or_effects_mismatch"
    )
    actual_predictions = [
        json.loads(line)
        for line in (output / "development-predictions.jsonl").read_text().splitlines()
    ]
    assert_json_equal(
        dev_predictions, actual_predictions, "v2_development_predictions_mismatch"
    )
    after_snapshot = _file_snapshot(observed_paths)
    if before_snapshot != after_snapshot:
        raise ValueError("v2_read_only_verification_changed_evidence_bytes_or_mtimes")
    return {
        "status": "verified_complete",
        "updates": 108,
        "current_runtime_production_C_equal": True,
        "D_executed": True,
        "historical_seed504_check": "descriptive only",
        "output_files": len([path for path in output.rglob("*") if path.is_file()]),
        "registered_inputs": len(registration["input_sha256"]),
        "read_only_bytes_and_mtimes_unchanged": True,
    }


def _file_snapshot(paths: set[Path]) -> dict[str, tuple[int, int, str]]:
    return {
        str(path): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            sha(path),
        )
        for path in sorted(paths, key=str)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tablebase", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.root.resolve()
    output = args.out.resolve()
    tablebase = (
        args.tablebase if args.tablebase.is_absolute() else root / args.tablebase
    )
    print(json.dumps(verify(root, output, tablebase), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
