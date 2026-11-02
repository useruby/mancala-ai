"""Independent read-only verifier for the seed549 controlled comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed549-controlled-training"
CORPUS = ROOT / "docs/data/seed549-matched-selfplay"
_TORCH_THREADS_CONFIGURED = False


def configure_torch_threads() -> None:
    """Configure one intra-op and one inter-op thread once per process."""
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _TORCH_THREADS_CONFIGURED = True


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def cgroup_limits() -> dict[str, str | None]:
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(
            line.split("::", 1)[1] for line in memberships if line.startswith("0::")
        )
        folder = Path("/sys/fs/cgroup") / relative.lstrip("/")
    except (OSError, StopIteration):
        return {
            name: None
            for name in ("cpu.max", "cpu.weight", "memory.max", "cpuset.cpus.effective")
        }
    result = {}
    for name in ("cpu.max", "cpu.weight", "memory.max", "cpuset.cpus.effective"):
        try:
            result[name] = (folder / name).read_text().strip()
        except OSError:
            result[name] = None
    return result


def file_snapshot(paths: set[Path]) -> dict[str, tuple[int, int, str]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, sha(path))
        for path in sorted(paths)
    }


def authenticate() -> dict[str, Any]:
    registration_path = OUT / "registration.json"
    registration = json.loads(registration_path.read_text())
    if json.loads((OUT / "execution-freeze.json").read_text()).get(
        "registration_sha256"
    ) != sha(registration_path):
        raise ValueError("registration_freeze_mismatch")
    if sha(OUT / "README.md") != registration["readme_sha256"]:
        raise ValueError("frozen_readme_mismatch")
    expected_sources = dict(registration["source_sha256"])
    parent_correction: Path | None = None
    parent_freeze: Path | None = None
    correction_number = 1
    while True:
        suffix = "" if correction_number == 1 else f"-{correction_number:02d}"
        correction_path = OUT / f"execution-correction{suffix}.json"
        freeze_correction_path = OUT / f"execution-freeze-correction{suffix}.json"
        if not correction_path.exists() and not freeze_correction_path.exists():
            break
        if not correction_path.is_file() or not freeze_correction_path.is_file():
            raise ValueError("verifier_source_correction_pair_incomplete")
        correction = json.loads(correction_path.read_text())
        relative = correction.get("source_path")
        if (
            correction.get("registration_sha256") != sha(registration_path)
            or relative not in expected_sources
            or correction.get("previous_sha256")
            not in {
                expected_sources.get(relative),
                json.loads(parent_correction.read_text())["corrected_sha256"]
                if parent_correction is not None
                else None,
            }
            or (
                parent_correction is not None
                and correction.get("parent_correction_sha256") != sha(parent_correction)
            )
        ):
            raise ValueError("verifier_source_correction_binding_mismatch")
        freeze_correction = json.loads(freeze_correction_path.read_text())
        if (
            freeze_correction.get("registration_sha256") != sha(registration_path)
            or freeze_correction.get("correction_sha256") != sha(correction_path)
            or (
                parent_freeze is not None
                and freeze_correction.get("parent_freeze_correction_sha256")
                != sha(parent_freeze)
            )
        ):
            raise ValueError("verifier_source_correction_freeze_mismatch")
        expected_sources[relative] = correction["corrected_sha256"]
        parent_correction = correction_path
        parent_freeze = freeze_correction_path
        correction_number += 1
    for relative, expected in expected_sources.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"frozen_source_mismatch:{relative}")
    for relative, expected in registration["input_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"frozen_input_mismatch:{relative}")
    if (
        sha(CORPUS / "shared-development-membership.jsonl")
        != registration["development"]["membership_sha256"]
    ):
        raise ValueError("development_membership_hash_mismatch")
    if (
        sha(ROOT / "docs/data/seed542-selfplay-4096/initializer.npz")
        != "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
    ):
        raise ValueError("initializer_identity_mismatch")
    if (sys.version, np.__version__, torch.__version__) != (
        registration["runtime"]["python"],
        registration["runtime"]["numpy"],
        registration["runtime"]["torch"],
    ):
        raise ValueError("runtime_mismatch")
    runtime = registration["runtime"]
    limits = runtime["cgroup_limits"]
    if (
        runtime["device"] != "cpu"
        or runtime["dtype"] != "float32"
        or runtime["cpu_affinity"] != [30]
        or runtime["nice"] != 19
        or runtime["address_space_rlimit"] != [17_179_869_184, 17_179_869_184]
        or runtime["environment"] != {key: "1" for key in runtime["environment"]}
        or limits["cpu.max"] != "25000 100000"
        or limits["cpu.weight"] != "1"
        or limits["memory.max"] != "4294967296"
        or limits["cpuset.cpus.effective"] not in {"30", None}
    ):
        raise ValueError("registered_resource_contract_invalid")
    if (
        torch.get_num_threads() != runtime["torch_threads"]
        or torch.get_num_interop_threads() != runtime["interop_threads"]
    ):
        raise ValueError("torch_thread_mismatch")
    if (
        sorted(os.sched_getaffinity(0)) != runtime["cpu_affinity"]
        or os.nice(0) != runtime["nice"]
    ):
        raise ValueError("resource_affinity_or_nice_mismatch")
    if list(resource.getrlimit(resource.RLIMIT_AS)) != runtime["address_space_rlimit"]:
        raise ValueError("address_space_limit_mismatch")
    if cgroup_limits() != runtime["cgroup_limits"]:
        raise ValueError("cgroup_limit_mismatch")
    env = {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "BLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    if env != runtime["environment"]:
        raise ValueError("thread_environment_mismatch")
    packages = sorted(
        (
            f"{d.metadata['Name']}=={d.version}"
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name")
        ),
        key=str.casefold,
    )
    if packages != runtime["packages"]:
        raise ValueError("dependency_set_mismatch")
    return registration


def read_arm(
    arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    identity_path = CORPUS / f"{arm}-training-row-identities.jsonl"
    identities = [
        json.loads(line)["row_id"] for line in identity_path.read_text().splitlines()
    ]
    paths = [CORPUS / arm / f"shard-{shard:03d}.jsonl" for shard in range(6)]
    x, p, v, _replay, weights = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    return x, p, v, weights, identities


def state_hash(net: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in net.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b"\0" + str(array.dtype).encode() + b"\0")
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


def array_evidence(value: np.ndarray | torch.Tensor) -> dict[str, Any]:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def require_json_equal(expected: Any, actual: Any, code: str) -> None:
    if stable(expected) != stable(actual):
        raise ValueError(code)


def validate_prediction_membership(
    prediction: dict[str, Any], expected: dict[str, Any], index: int, endpoint: str
) -> None:
    if prediction.get("occurrence") != index or any(
        stable(prediction.get(key)) != stable(value) for key, value in expected.items()
    ):
        raise ValueError(
            f"development_prediction_membership_mismatch:{endpoint}:{index}"
        )


def optimizer_hash(opt: torch.optim.Optimizer, net: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, parameter in net.named_parameters():
        h.update(name.encode() + b"\0")
        for key, value in sorted(opt.state.get(parameter, {}).items()):
            h.update(key.encode() + b"\0")
            if isinstance(value, torch.Tensor):
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(repr(value).encode())
    return h.hexdigest()


def grad_hash(net: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, parameter in net.named_parameters():
        value = (
            parameter.grad.detach().cpu().contiguous().numpy()
            if parameter.grad is not None
            else np.zeros(tuple(parameter.shape), dtype=np.float32)
        )
        h.update(name.encode() + b"\0")
        h.update(value.tobytes())
    return h.hexdigest()


def replay(arm: str) -> None:
    data = read_arm(arm)
    x, p, v, weights, identities = data
    path = CORPUS / f"{arm}-training-schedule.json"
    schedule = json.loads(path.read_text())
    if schedule["updates"] != 108 or schedule["exposures"] != 54896:
        raise ValueError(f"schedule_header_mismatch:{arm}")
    batches = schedule["batches"]
    trace_path = OUT / f"{arm}-updates.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if len(batches) != 108 or len(records) != 108:
        raise ValueError(f"update_count_mismatch:{arm}")
    net = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        net, ROOT / "docs/data/seed542-selfplay-4096/initializer.npz"
    )
    opt = torch.optim.Adam(
        net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    boundaries = {27, 54, 81, 108}
    for number, (batch, record) in enumerate(zip(batches, records, strict=True), 1):
        idx = batch["row_occurrence_indices"]
        if (
            batch["update"] != number
            or record["update"] != number
            or record["row_occurrence_ids"] != [identities[i] for i in idx]
        ):
            raise ValueError(f"schedule_to_row_mapping_mismatch:{arm}:{number}")
        if arm == "control" and (
            record.get("production_equivalent") is not True
            or record.get("production_parameter_state_after_sha256")
            != record.get("parameter_state_after_sha256")
        ):
            raise ValueError(f"production_equivalence_gate_missing:{number}")
        index = np.asarray(idx, dtype=np.int64)
        tx, tp, tv, tw = (
            torch.from_numpy(array[index]) for array in (x, p, v, weights)
        )
        mask = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[index]))
        before = state_hash(net)
        adam_before = optimizer_hash(opt, net)
        logits, predicted = net(tx)
        # Reconstruct the objective without calling runner/publisher loss functions.
        logp = torch.log_softmax(logits.masked_fill(mask <= 0, -1e9), dim=1)
        per_row_ce = -(tp * logp).sum(dim=1)
        denominator = tw.sum()
        policy = (
            (per_row_ce * tw).sum() / denominator
            if float(denominator)
            else torch.zeros((), dtype=torch.float32)
        )
        value = (
            torch.nn.functional.smooth_l1_loss(
                predicted, tv, beta=1.0, reduction="none"
            )
            .reshape(-1)
            .mean()
        )
        total = policy + 0.5 * value
        opt.zero_grad(set_to_none=True)
        total.backward()
        raw = float(
            torch.sqrt(
                sum(
                    torch.sum(param.grad.detach() ** 2)
                    for param in net.parameters()
                    if param.grad is not None
                )
            ).item()
        )
        raw_gradient = grad_hash(net)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        clipped_gradient = grad_hash(net)
        post = float(
            torch.sqrt(
                sum(
                    torch.sum(param.grad.detach() ** 2)
                    for param in net.parameters()
                    if param.grad is not None
                )
            ).item()
        )
        opt.step()
        after = state_hash(net)
        adam_after = optimizer_hash(opt, net)
        expected = {
            "parameter_state_before_sha256": before,
            "parameter_state_after_sha256": after,
            "policy_loss": float(policy.detach()),
            "value_loss": float(value.detach()),
            "objective": float(total.detach()),
            "gradient_norm_independently_measured": raw,
            "gradient_norm_postclip": post,
            "batch_size": len(idx),
            "batch_identity_sha256": hashlib.sha256(
                np.asarray(idx, dtype="<u8").tobytes()
            ).hexdigest(),
            "optimizer_state_before_sha256": adam_before,
            "optimizer_state_after_sha256": adam_after,
            "raw_gradient_sha256": raw_gradient,
            "clipped_gradient_sha256": clipped_gradient,
            "clip_scale": min(1.0, 1.0 / raw) if raw else 1.0,
            "tensors": {
                "input": array_evidence(tx),
                "policy_target": array_evidence(tp),
                "value_target": array_evidence(tv),
                "policy_weight": array_evidence(tw),
                "legal_mask": array_evidence(mask),
            },
        }
        for key, value_ in expected.items():
            matches = record.get(key) == value_
            if key in {
                "gradient_norm_independently_measured",
                "gradient_norm_postclip",
                "clip_scale",
            }:
                matches = np.isclose(record.get(key), value_, rtol=1e-6, atol=1e-7)
            if not matches:
                raise ValueError(f"training_replay_mismatch:{arm}:{number}:{key}")
        if number in boundaries:
            checkpoint = OUT / f"{arm}-update-{number:03d}.npz"
            with np.load(checkpoint) as saved:
                expected_arrays = train.checkpoint_from_model(net)
                if set(saved.files) != set(expected_arrays) or any(
                    not np.array_equal(saved[k], expected_arrays[k])
                    for k in expected_arrays
                ):
                    raise ValueError(f"checkpoint_mismatch:{arm}:{number}")


def _verify_authenticated(root: Path, output: Path) -> dict[str, Any]:
    global ROOT, OUT, CORPUS
    ROOT, OUT = root.resolve(), output.resolve()
    CORPUS = ROOT / "docs/data/seed549-matched-selfplay"
    registration = authenticate()
    observed = {ROOT / p for p in registration["input_sha256"]}
    observed |= {p for p in OUT.rglob("*") if p.is_file()}
    before = file_snapshot(observed)
    for arm in ("control", "treatment"):
        replay(arm)
    summary = json.loads((OUT / "run-summary.json").read_text())
    if summary.get("candidate") != "T-final at update 108 only":
        raise ValueError("candidate_designation_mismatch")
    for arm, label in (("control", "control"), ("treatment", "treatment")):
        arm_summary = summary.get(label, {})
        trace_path = OUT / f"{arm}-updates.jsonl"
        expected_checkpoints = {
            str(update): sha(OUT / f"{arm}-update-{update:03d}.npz")
            for update in (27, 54, 81, 108)
        }
        if (
            arm_summary.get("rows") != (13_818 if arm == "control" else 13_849)
            or arm_summary.get("updates") != 108
            or arm_summary.get("exposures") != 54_896
            or arm_summary.get("trace_sha256") != sha(trace_path)
            or arm_summary.get("checkpoints") != expected_checkpoints
        ):
            raise ValueError(f"run_summary_evidence_mismatch:{arm}")
    # Independently reconstruct metrics and all reported effects from the
    # occurrence-bound prediction evidence; publisher calculations are not called.
    published = json.loads((OUT / "development-results.json").read_text())
    rows = [
        json.loads(line)
        for line in (OUT / "development-predictions.jsonl").read_text().splitlines()
    ]
    if len(rows) != 3 * 4738:
        raise ValueError("development_prediction_row_count_mismatch")
    membership = [
        json.loads(line)
        for line in (CORPUS / "shared-development-membership.jsonl")
        .read_text()
        .splitlines()
        if json.loads(line)["included"]
    ]
    if len(membership) != 4738:
        raise ValueError("retained_development_membership_count_mismatch")
    development_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for arm in ("control", "treatment"):
        development_rows[arm] = {}
        for shard in (6, 7):
            path = CORPUS / arm / f"shard-{shard:03d}.jsonl"
            development_rows[arm][path.as_posix()] = [
                json.loads(line) for line in path.read_text().splitlines()
            ]
    expected_dev: list[dict[str, Any]] = []
    dev_inputs: list[np.ndarray] = []
    for item in membership:
        source = item["source"]
        arm = source["arm"]
        row = development_rows[arm][(ROOT / source["path"]).as_posix()][
            int(source["row_number"])
        ]
        state = np.asarray(row["state"], dtype=np.float32)
        policy = np.asarray(row["policy"], dtype=np.float32)
        target_value = float(np.float32(row["value"]))
        if (
            hashlib.sha256(state.tobytes()).hexdigest()
            != item["production_float32_sha256"]
        ):
            raise ValueError("development_float32_membership_mismatch")
        expected_dev.append(
            {
                "source_evaluator": item["source_evaluator"],
                "cohort": item["stone_bucket"],
                "float32_input_sha256": item["production_float32_sha256"],
                "target_policy": policy.tolist(),
                "target_actual_value": target_value,
                "legal_mask": train.legal_mask_for_encoded_state(state)
                .astype(bool)
                .tolist(),
            }
        )
        dev_inputs.append(state)
    for endpoint in ("initializer", "C-final", "T-final"):
        endpoint_rows = [row for row in rows if row["endpoint"] == endpoint]
        checkpoint = (
            ROOT / "docs/data/seed542-selfplay-4096/initializer.npz"
            if endpoint == "initializer"
            else OUT
            / (
                "control-update-108.npz"
                if endpoint == "C-final"
                else "treatment-update-108.npz"
            )
        )
        endpoint_model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(endpoint_model, checkpoint)
        endpoint_model.eval()
        endpoint_x = np.asarray(dev_inputs, dtype=np.float32)
        endpoint_legal = train.legal_mask_matrix_for_encoded_states(endpoint_x).astype(
            bool
        )
        with torch.no_grad():
            replay_logits, replay_values = endpoint_model(torch.from_numpy(endpoint_x))
        replay_logits_np = replay_logits.numpy().astype(np.float64)
        replay_logits_np[~endpoint_legal] = -1e9
        replay_probabilities = np.exp(
            replay_logits_np - np.logaddexp.reduce(replay_logits_np, axis=1)[:, None]
        )
        replay_values_np = replay_values.numpy().reshape(-1).astype(np.float64)
        for index, (prediction, expected) in enumerate(
            zip(endpoint_rows, expected_dev, strict=True)
        ):
            validate_prediction_membership(prediction, expected, index, endpoint)
            if (
                not np.array_equal(
                    np.asarray(
                        prediction["policy_logits_legal_masked"], dtype=np.float64
                    ),
                    replay_logits_np[index],
                )
                or not np.array_equal(
                    np.asarray(
                        prediction["policy_probabilities_legal_masked"],
                        dtype=np.float64,
                    ),
                    replay_probabilities[index],
                )
                or float(prediction["value_prediction"])
                != float(replay_values_np[index])
            ):
                raise ValueError(
                    f"endpoint_prediction_replay_mismatch:{endpoint}:{index}"
                )
            logits = np.asarray(
                prediction["policy_logits_legal_masked"], dtype=np.float64
            )
            probability = np.exp(logits - np.logaddexp.reduce(logits))
            if not np.allclose(
                probability,
                prediction["policy_probabilities_legal_masked"],
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"development_probability_evidence_mismatch:{endpoint}:{index}"
                )
    reconstructed: dict[str, Any] = {
        "weighting_definitions": {
            "exposure": "mean loss over retained occurrences",
            "equal_input": "mean occurrence loss within each exact float32 input group, then mean equally across groups",
        },
        "metrics": {},
        "effects": {},
    }
    cohorts = ("pooled", ">32", "17-32", "<=16")
    for endpoint in ("initializer", "C-final", "T-final"):
        selected = [r for r in rows if r["endpoint"] == endpoint]
        if [r["occurrence"] for r in selected] != list(range(4738)):
            raise ValueError(f"development_order_mismatch:{endpoint}")
        ce: list[float] = []
        mse: list[float] = []
        for row in selected:
            logits = np.asarray(row["policy_logits_legal_masked"], dtype=np.float64)
            target = np.asarray(row["target_policy"], dtype=np.float64)
            ce.append(float(-(target * (logits - np.logaddexp.reduce(logits))).sum()))
            mse.append(
                (float(row["value_prediction"]) - float(row["target_actual_value"]))
                ** 2
            )
        endpoint_cohorts: dict[str, Any] = {}
        for source in ("control", "treatment", "combined"):
            endpoint_cohorts[source] = {}
            for cohort in cohorts:
                indexes = [
                    i
                    for i, row in enumerate(selected)
                    if (source == "combined" or row["source_evaluator"] == source)
                    and (cohort == "pooled" or row["cohort"] == cohort)
                ]
                by_input: dict[str, list[int]] = defaultdict(list)
                for i in indexes:
                    by_input[selected[i]["float32_input_sha256"]].append(i)
                endpoint_cohorts[source][cohort] = {
                    "occurrences": len(indexes),
                    "distinct_float32_inputs": len(by_input),
                    "exposure": {
                        "policy_ce": float(np.mean([ce[i] for i in indexes])),
                        "actual_value_mse": float(np.mean([mse[i] for i in indexes])),
                    }
                    if indexes
                    else None,
                    "equal_input": {
                        "policy_ce": float(
                            np.mean(
                                [
                                    np.mean([ce[i] for i in group])
                                    for group in by_input.values()
                                ]
                            )
                        ),
                        "actual_value_mse": float(
                            np.mean(
                                [
                                    np.mean([mse[i] for i in group])
                                    for group in by_input.values()
                                ]
                            )
                        ),
                    }
                    if by_input
                    else None,
                }
        checkpoint = (
            ROOT / "docs/data/seed542-selfplay-4096/initializer.npz"
            if endpoint == "initializer"
            else OUT
            / (
                "control-update-108.npz"
                if endpoint == "C-final"
                else "treatment-update-108.npz"
            )
        )
        reconstructed["metrics"][endpoint] = {
            "checkpoint_sha256": sha(checkpoint),
            "cohorts": endpoint_cohorts,
        }
    for endpoint, baseline in (
        ("T-final", "C-final"),
        ("C-final", "initializer"),
        ("T-final", "initializer"),
    ):
        label = f"{endpoint}_minus_{baseline}"
        reconstructed["effects"][label] = {}
        for source in ("control", "treatment", "combined"):
            reconstructed["effects"][label][source] = {}
            for cohort in cohorts:
                reconstructed["effects"][label][source][cohort] = {}
                for weighting in ("exposure", "equal_input"):
                    left = reconstructed["metrics"][endpoint]["cohorts"][source][
                        cohort
                    ][weighting]
                    right = reconstructed["metrics"][baseline]["cohorts"][source][
                        cohort
                    ][weighting]
                    reconstructed["effects"][label][source][cohort][weighting] = (
                        None
                        if left is None or right is None
                        else {
                            metric: left[metric] - right[metric]
                            for metric in ("policy_ce", "actual_value_mse")
                        }
                    )
    require_json_equal(
        reconstructed, published, "development_metrics_or_effects_mismatch"
    )
    after = file_snapshot(observed)
    if before != after:
        raise ValueError("read_only_verification_changed_bytes_or_mtimes")
    return {
        "status": "verified",
        "training_updates_per_arm": 108,
        "development_occurrences": 4738,
        "predictions": len(rows),
        "bytes_and_mtimes_unchanged": True,
    }


def verify(root: Path = ROOT, output: Path = OUT) -> dict[str, Any]:
    """Verify evidence read-only, checking bytes and mtimes even on rejection."""
    configure_torch_threads()
    global ROOT, OUT, CORPUS
    ROOT, OUT = root.resolve(), output.resolve()
    CORPUS = ROOT / "docs/data/seed549-matched-selfplay"
    known = {
        ROOT / "docs/data/seed542-selfplay-4096/initializer.npz",
        *(path for path in CORPUS.rglob("*") if path.is_file()),
        *(path for path in OUT.rglob("*") if path.is_file()),
    }
    registration_path = OUT / "registration.json"
    if registration_path.is_file():
        try:
            registration = json.loads(registration_path.read_text())
            known.update(ROOT / name for name in registration.get("source_sha256", {}))
            known.update(ROOT / name for name in registration.get("input_sha256", {}))
        except (OSError, json.JSONDecodeError):
            pass
    before = file_snapshot(known)
    try:
        return _verify_authenticated(ROOT, OUT)
    finally:
        if file_snapshot(known) != before:
            raise ValueError("read_only_verification_changed_bytes_or_mtimes")


def main() -> None:
    configure_torch_threads()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--bind-verifier-correction", action="store_true")
    parser.add_argument("--correction-reason", default=None)
    parser.add_argument(
        "--correction-source",
        default="ml/alphazero_lite/verify_seed549_training.py",
    )
    args = parser.parse_args()
    if args.bind_verifier_correction:
        args.out.mkdir(parents=True, exist_ok=True)
        relative = args.correction_source
        registration_path = args.out / "registration.json"
        registration = json.loads(registration_path.read_text())
        if relative not in registration["source_sha256"]:
            raise ValueError("correction_source_not_in_original_freeze")
        prior = registration["source_sha256"][relative]
        parent_correction = None
        parent_freeze = None
        number = 1
        while True:
            suffix = "" if number == 1 else f"-{number:02d}"
            candidate = args.out / f"execution-correction{suffix}.json"
            freeze_candidate = args.out / f"execution-freeze-correction{suffix}.json"
            if not candidate.exists() and not freeze_candidate.exists():
                break
            current = json.loads(candidate.read_text())
            prior = current["corrected_sha256"]
            parent_correction = candidate
            parent_freeze = freeze_candidate
            number += 1
        suffix = "" if number == 1 else f"-{number:02d}"
        correction_path = args.out / f"execution-correction{suffix}.json"
        freeze_path = args.out / f"execution-freeze-correction{suffix}.json"
        corrected = sha((args.root / relative).resolve())
        correction = {
            "schema": "seed549-training-verifier-source-correction-v1",
            "registration_sha256": sha(registration_path),
            "source_path": relative,
            "previous_sha256": prior,
            "corrected_sha256": corrected,
            "reason": args.correction_reason
            or "narrowly correct independent verifier after preserving the prior verifier failure",
        }
        if parent_correction is not None:
            correction["parent_correction_sha256"] = sha(parent_correction)
        correction_path.write_text(
            json.dumps(correction, indent=2, sort_keys=True) + "\n"
        )
        freeze_value = {
            "registration_sha256": sha(registration_path),
            "correction_sha256": sha(correction_path),
        }
        if parent_freeze is not None:
            freeze_value["parent_freeze_correction_sha256"] = sha(parent_freeze)
        freeze_path.write_text(
            json.dumps(freeze_value, indent=2, sort_keys=True) + "\n"
        )
        return
    print(json.dumps(verify(args.root, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
