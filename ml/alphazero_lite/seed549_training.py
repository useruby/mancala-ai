"""Controlled seed549 training comparison runner and evidence publisher."""

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

from ml.alphazero_lite import run_seed542_fixed_update_cd as reference
from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed549-matched-selfplay"
OUT = ROOT / "docs/data/seed549-controlled-training"
INITIALIZER = ROOT / "docs/data/seed542-selfplay-4096/initializer.npz"
BOUNDARIES = {27, 54, 81, 108}
ARMS = ("control", "treatment")
BUCKETS = (">32", "17-32", "<=16")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
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


def array_record(value: np.ndarray | torch.Tensor) -> dict[str, Any]:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    array = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def load_arm(
    arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if arm not in ARMS:
        raise ValueError(f"unknown_arm:{arm}")
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
    if len(identities) != len(x) or len(set(identities)) != len(identities):
        raise ValueError(f"training_occurrence_identity_mismatch:{arm}")
    expected = 13_818 if arm == "control" else 13_849
    if len(identities) != expected:
        raise ValueError(f"training_occurrence_count_mismatch:{arm}:{len(identities)}")
    return x, p, v, weights, identities


def load_schedule(arm: str) -> list[dict[str, Any]]:
    path = CORPUS / f"{arm}-training-schedule.json"
    schedule = json.loads(path.read_text())
    batches = schedule["batches"]
    sizes = [512] * 26 + [412]
    expected = sizes * 4
    if (
        schedule.get("updates") != 108
        or schedule.get("exposures") != 54_896
        or len(batches) != 108
    ):
        raise ValueError(f"schedule_shape_mismatch:{arm}")
    cursor = 0
    for number, (batch, size) in enumerate(zip(batches, expected, strict=True), 1):
        indices = batch["row_occurrence_indices"]
        if (
            batch.get("update") != number
            or len(indices) != size
            or batch.get("start_exposure") != cursor
            or batch.get("end_exposure_exclusive") != cursor + size
        ):
            raise ValueError(f"schedule_batch_mapping_mismatch:{arm}:{number}")
        if any(
            not isinstance(i, int)
            or i < 0
            or i >= (13_818 if arm == "control" else 13_849)
            for i in indices
        ):
            raise ValueError(f"schedule_row_index_out_of_range:{arm}:{number}")
        cursor += size
    return batches


def model() -> train.PolicyValueNet:
    result = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(result, INITIALIZER)
    return result.cpu()


def optimizer_state_id(
    optimizer: torch.optim.Optimizer, model_: torch.nn.Module
) -> str:
    digest = hashlib.sha256()
    for name, parameter in model_.named_parameters():
        digest.update(name.encode() + b"\0")
        for key, value in sorted(optimizer.state.get(parameter, {}).items()):
            digest.update(key.encode() + b"\0")
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def gradient_identity(model_: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model_.named_parameters():
        gradient = (
            parameter.grad.detach().cpu().contiguous().numpy()
            if parameter.grad is not None
            else np.zeros(tuple(parameter.shape), dtype=np.float32)
        )
        digest.update(name.encode() + b"\0")
        digest.update(gradient.tobytes())
    return digest.hexdigest()


def update(
    model_: train.PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    arrays: tuple[np.ndarray, ...],
    indices: list[int],
) -> dict[str, Any]:
    x, p, v, weights = arrays
    ix = np.asarray(indices, dtype=np.int64)
    bx, bp, bv, bw = (torch.from_numpy(item[ix]) for item in (x, p, v, weights))
    legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ix]))
    model_.train()
    before = reference.state_digest(model_)
    optimizer_before = optimizer_state_id(optimizer, model_)
    logits, prediction = model_(bx)
    policy_loss = train.weighted_policy_loss(
        train.compute_policy_cross_entropy(logits.masked_fill(legal <= 0, -1e9), bp), bw
    )
    value_loss = train.compute_value_loss_vector(
        prediction, bv, value_loss="huber", huber_delta=1.0
    ).mean()
    objective = policy_loss + 0.5 * value_loss
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    raw_norm = float(
        torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(parameter.grad.detach())
                    for parameter in model_.parameters()
                    if parameter.grad is not None
                ]
            )
        ).item()
    )
    raw_gradient_identity = gradient_identity(model_)
    clip_return = torch.nn.utils.clip_grad_norm_(model_.parameters(), 1.0)
    post_norm = min(raw_norm, 1.0)
    optimizer.step()
    return {
        "batch_indices": indices,
        "batch_size": len(indices),
        "batch_identity_sha256": hashlib.sha256(
            np.asarray(indices, dtype="<u8").tobytes()
        ).hexdigest(),
        "tensors": {
            "input": array_record(bx),
            "policy_target": array_record(bp),
            "value_target": array_record(bv),
            "policy_weight": array_record(bw),
            "legal_mask": array_record(legal),
        },
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "objective": float(objective.detach()),
        "gradient_norm_preclip": float(clip_return),
        "gradient_norm_independently_measured": raw_norm,
        "gradient_norm_postclip": post_norm,
        "raw_gradient_sha256": raw_gradient_identity,
        "clipped_gradient_sha256": gradient_identity(model_),
        "clip_scale": min(1.0, 1.0 / raw_norm) if raw_norm else 1.0,
        "clip_active": raw_norm > 1.0,
        "optimizer_state_before_sha256": optimizer_before,
        "optimizer_state_after_sha256": optimizer_state_id(optimizer, model_),
        "parameter_state_before_sha256": before,
        "parameter_state_after_sha256": reference.state_digest(model_),
    }


def run_arm(arm: str, *, production: bool = False) -> dict[str, Any]:
    x, p, v, weights, ids = load_arm(arm)
    batches = load_schedule(arm)
    arrays = (x, p, v, weights)
    model_ = model()
    optimizer = torch.optim.Adam(
        model_.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    production_model = model() if arm == "control" else None
    production_optimizer = (
        torch.optim.Adam(
            production_model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        if production_model is not None
        else None
    )
    trace = OUT / f"{arm}-updates.jsonl"
    checkpoints: dict[str, str] = {}
    with trace.open("x", encoding="utf-8") as stream:
        for batch in batches:
            idx = batch["row_occurrence_indices"]
            record = update(model_, optimizer, arrays, idx)
            record["update"] = batch["update"]
            record["row_occurrence_ids"] = [ids[i] for i in idx]
            if production_model is not None and production_optimizer is not None:
                # Exercise the production epoch/update primitive on precisely the
                # registered batch. Identity ordering is forced only for this
                # one-batch invocation; no target or loss implementation is copied.
                ix = np.asarray(idx, dtype=np.int64)
                original_randperm = torch.randperm
                torch.randperm = lambda n, *a, **kw: torch.arange(
                    n, device=kw.get("device", "cpu")
                )
                try:
                    train.train_one_epoch(
                        model=production_model,
                        optimizer=production_optimizer,
                        compact_x=x[ix],
                        compact_p=p[ix],
                        compact_v=v[ix],
                        compact_policy_loss_weights=weights[ix],
                        replay_indexes=np.arange(len(ix), dtype=np.int64),
                        batch_size=len(ix),
                        device=torch.device("cpu"),
                        value_loss_weight=0.5,
                        value_loss="huber",
                        huber_delta=1.0,
                        grad_clip=1.0,
                        epoch=batch["update"],
                        max_optimizer_updates=1,
                    )
                finally:
                    torch.randperm = original_randperm
                production_digest = reference.state_digest(production_model)
                record["production_parameter_state_after_sha256"] = production_digest
                record["production_equivalent"] = (
                    production_digest == record["parameter_state_after_sha256"]
                )
                if not record["production_equivalent"]:
                    raise RuntimeError(
                        f"control_production_equivalence_failed:{batch['update']}"
                    )
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if batch["update"] in BOUNDARIES:
                checkpoint = OUT / f"{arm}-update-{batch['update']:03d}.npz"
                np.savez(checkpoint, **train.checkpoint_from_model(model_))
                checkpoints[str(batch["update"])] = sha(checkpoint)
    return {
        "arm": arm,
        "rows": len(x),
        "updates": 108,
        "exposures": 54_896,
        "trace_sha256": sha(trace),
        "checkpoints": checkpoints,
    }


def evaluate() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dev_path = CORPUS / "shared-development-membership.jsonl"
    membership = [json.loads(line) for line in dev_path.read_text().splitlines()]
    retained = [row for row in membership if row["included"]]
    if len(retained) != 4_738:
        raise ValueError(f"retained_development_count_mismatch:{len(retained)}")
    all_rows = {}
    for arm in ARMS:
        all_rows[arm] = {}
        for shard in (6, 7):
            path = CORPUS / arm / f"shard-{shard:03d}.jsonl"
            all_rows[arm][path.as_posix()] = [
                json.loads(line) for line in path.read_text().splitlines()
            ]
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in retained:
        rowid = item["source"]
        arm = item["source"]["arm"]
        source_path = (ROOT / rowid["path"]).as_posix()
        selected.append((item, all_rows[arm][source_path][int(rowid["row_number"])]))
    inputs = np.asarray([row["state"] for _, row in selected], dtype=np.float32)
    policies = np.asarray([row["policy"] for _, row in selected], dtype=np.float32)
    targets = np.asarray([row["value"] for _, row in selected], dtype=np.float32)
    sources = [item["source_evaluator"] for item, _ in selected]
    stones = np.rint(inputs[:, :12].sum(axis=1) * 48).astype(int)
    buckets = np.where(stones > 32, ">32", np.where(stones >= 17, "17-32", "<=16"))
    endpoint_paths = {
        "initializer": INITIALIZER,
        "C-final": OUT / "control-update-108.npz",
        "T-final": OUT / "treatment-update-108.npz",
    }
    output: dict[str, Any] = {
        "weighting_definitions": {
            "exposure": "mean loss over retained occurrences",
            "equal_input": "mean occurrence loss within each exact float32 input group, then mean equally across groups",
        },
        "metrics": {},
        "effects": {},
    }
    evidence = []
    for endpoint, checkpoint in endpoint_paths.items():
        net = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(net, checkpoint)
        net.eval()
        legal = train.legal_mask_matrix_for_encoded_states(inputs).astype(bool)
        with torch.no_grad():
            logits_t, values_t = net(torch.from_numpy(inputs))
        logits = logits_t.numpy().astype(np.float64)
        logits[~legal] = -1e9
        logp = logits - np.logaddexp.reduce(logits, axis=1)[:, None]
        probs = np.exp(logp)
        pred = values_t.numpy().reshape(-1).astype(np.float64)
        ce = -(policies.astype(np.float64) * logp).sum(axis=1)
        mse = (pred - targets.astype(np.float64)) ** 2
        endpoint_metrics = {}
        for source in (*ARMS, "combined"):
            source_mask = (
                np.ones(len(selected), dtype=bool)
                if source == "combined"
                else np.asarray(sources) == source
            )
            endpoint_metrics[source] = {}
            for cohort in ("pooled", *BUCKETS):
                mask = (
                    source_mask
                    if cohort == "pooled"
                    else source_mask & (buckets == cohort)
                )
                indices = np.flatnonzero(mask)
                groups: dict[bytes, list[int]] = defaultdict(list)
                for i in indices:
                    groups[inputs[i].tobytes()].append(int(i))
                endpoint_metrics[source][cohort] = {
                    "occurrences": len(indices),
                    "distinct_float32_inputs": len(groups),
                    "exposure": {
                        "policy_ce": float(ce[indices].mean()),
                        "actual_value_mse": float(mse[indices].mean()),
                    }
                    if len(indices)
                    else None,
                    "equal_input": {
                        "policy_ce": float(
                            np.mean([ce[g].mean() for g in groups.values()])
                        ),
                        "actual_value_mse": float(
                            np.mean([mse[g].mean() for g in groups.values()])
                        ),
                    }
                    if groups
                    else None,
                }
        output["metrics"][endpoint] = {
            "checkpoint_sha256": sha(checkpoint),
            "cohorts": endpoint_metrics,
        }
        evidence.extend(
            {
                "endpoint": endpoint,
                "occurrence": i,
                "source_evaluator": sources[i],
                "cohort": str(buckets[i]),
                "float32_input_sha256": hashlib.sha256(inputs[i].tobytes()).hexdigest(),
                "target_policy": policies[i].tolist(),
                "target_actual_value": float(targets[i]),
                "legal_mask": legal[i].tolist(),
                "policy_logits_legal_masked": logits[i].tolist(),
                "policy_probabilities_legal_masked": probs[i].tolist(),
                "value_prediction": float(pred[i]),
            }
            for i in range(len(selected))
        )
    for endpoint, base in (
        ("T-final", "C-final"),
        ("C-final", "initializer"),
        ("T-final", "initializer"),
    ):
        label = f"{endpoint}_minus_{base}"
        output["effects"][label] = {}
        for source in (*ARMS, "combined"):
            output["effects"][label][source] = {}
            for cohort in ("pooled", *BUCKETS):
                output["effects"][label][source][cohort] = {}
                for weighting in ("exposure", "equal_input"):
                    a = output["metrics"][endpoint]["cohorts"][source][cohort][
                        weighting
                    ]
                    b = output["metrics"][base]["cohorts"][source][cohort][weighting]
                    output["effects"][label][source][cohort][weighting] = (
                        None
                        if a is None or b is None
                        else {
                            metric: a[metric] - b[metric]
                            for metric in ("policy_ce", "actual_value_mse")
                        }
                    )
    return output, evidence


def publish() -> None:
    result, evidence = evaluate()
    (OUT / "development-results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (OUT / "development-predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in evidence:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def freeze() -> None:
    if OUT.exists() and any(path.name != "README.md" for path in OUT.iterdir()):
        raise FileExistsError("training_freeze_output_not_empty")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source_names = (
        "ml/alphazero_lite/seed549_training.py",
        "ml/alphazero_lite/verify_seed549_training.py",
        "ml/alphazero_lite/test_seed549_training.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/run_seed542_fixed_update_cd.py",
        "ml/alphazero_lite/input_encodings.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/exact_root_policy_targets.py",
        "ml/alphazero_lite/checkpoint_phase_selection.py",
        "ml/alphazero_lite/seed549_corpus.py",
        "ml/alphazero_lite/seed549_derivatives.py",
        "ml/alphazero_lite/selfplay_evaluator_pilot.py",
        "ml/alphazero_lite/verify_selfplay_evaluator_pilot.py",
    )
    input_paths = [INITIALIZER, *(p for p in CORPUS.rglob("*") if p.is_file())]
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(input_paths)}
    packages = sorted(
        (
            f"{d.metadata['Name']}=={d.version}"
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name")
        ),
        key=str.casefold,
    )
    registration = {
        "schema": "seed549-controlled-training-v1",
        "claim": "source self-play evaluator comparison; evaluator effects include states, targets, outcomes, and occurrence frequencies; not policy-target quality alone",
        "candidate": "T-final update 108 only",
        "initializer_sha256": sha(INITIALIZER),
        "input_sha256": input_hashes,
        "readme_sha256": sha(OUT / "README.md"),
        "source_sha256": {name: sha(ROOT / name) for name in source_names},
        "schedules": {
            arm: sha(CORPUS / f"{arm}-training-schedule.json") for arm in ARMS
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "packages": packages,
            "device": "cpu",
            "dtype": "float32",
            "torch_threads": 1,
            "interop_threads": 1,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "nice": os.nice(0),
            "address_space_rlimit": list(resource.getrlimit(resource.RLIMIT_AS)),
            "cgroup_limits": cgroup_limits(),
            "environment": {
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "BLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "training": {
            "architecture": "residual_v3/kalah_v3",
            "hidden": [96, 3],
            "inputs": 27,
            "updates": 108,
            "exposures": 54896,
            "batch_sizes": [512] * 26 + [412],
            "repetitions": 4,
            "objective": "legal policy CE + 0.5 * Huber(value, delta=1)",
            "optimizer": {
                "name": "Adam",
                "lr": 0.0001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "scheduler": "none",
            },
            "gradient_clip": 1.0,
        },
        "development": {
            "membership_sha256": sha(CORPUS / "shared-development-membership.jsonl"),
            "rows": 4738,
            "cohort_order": ["pooled", ">32", "17-32", "<=16"],
            "evaluation_chunking": "single ordered float32 batch; 4738 occurrences",
            "metric_formulas": "registered by seed549_training.evaluate",
        },
        "resource_scope": "CPU-only; one torch intra/inter-op thread; same process and environment for both arms",
    }
    runtime = registration["runtime"]
    limits = runtime["cgroup_limits"]
    if (
        runtime["cpu_affinity"] != [30]
        or runtime["nice"] != 19
        or runtime["address_space_rlimit"] != [17_179_869_184, 17_179_869_184]
        or runtime["environment"] != {key: "1" for key in runtime["environment"]}
        or limits["cpu.max"] != "25000 100000"
        or limits["cpu.weight"] != "1"
        or limits["memory.max"] != "4294967296"
        or limits["cpuset.cpus.effective"] not in {"30", None}
    ):
        raise ValueError("registered_training_resource_contract_mismatch")
    path = OUT / "registration.json"
    path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    (OUT / "execution-freeze.json").write_text(
        json.dumps({"registration_sha256": sha(path)}, indent=2, sort_keys=True) + "\n"
    )


def preflight() -> None:
    registration_path = OUT / "registration.json"
    registration = json.loads(registration_path.read_text())
    frozen = json.loads((OUT / "execution-freeze.json").read_text())
    if frozen.get("registration_sha256") != sha(registration_path):
        raise ValueError("registration_freeze_mismatch")
    if sha(OUT / "README.md") != registration["readme_sha256"]:
        raise ValueError("frozen_readme_mismatch")
    for relative, expected in registration["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"frozen_source_mismatch:{relative}")
    for relative, expected in registration["input_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"frozen_input_mismatch:{relative}")
    if (sys.version, np.__version__, torch.__version__) != (
        registration["runtime"]["python"],
        registration["runtime"]["numpy"],
        registration["runtime"]["torch"],
    ):
        raise ValueError("frozen_runtime_mismatch")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("frozen_torch_thread_mismatch")
    if registration["runtime"]["cpu_affinity"] != sorted(os.sched_getaffinity(0)):
        raise ValueError("frozen_cpu_affinity_mismatch")
    if registration["runtime"]["nice"] != os.nice(0):
        raise ValueError("frozen_nice_mismatch")
    if registration["runtime"]["address_space_rlimit"] != list(
        resource.getrlimit(resource.RLIMIT_AS)
    ):
        raise ValueError("frozen_address_space_limit_mismatch")
    if registration["runtime"]["cgroup_limits"] != cgroup_limits():
        raise ValueError("frozen_cgroup_limit_mismatch")
    if registration["runtime"]["environment"] != {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "BLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }:
        raise ValueError("frozen_thread_environment_mismatch")
    if registration["runtime"]["packages"] != sorted(
        (
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ),
        key=str.casefold,
    ):
        raise ValueError("frozen_dependency_set_mismatch")


def execute() -> None:
    preflight()
    unexpected = [
        p.name
        for p in OUT.iterdir()
        if p.name not in {"registration.json", "execution-freeze.json", "README.md"}
    ]
    if unexpected:
        raise FileExistsError(
            "seed549_training_attempt_already_exists:" + ",".join(unexpected)
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    c = run_arm("control")
    # C/T use identical runtime; control output is the explicit gate to continue.
    t = run_arm("treatment")
    (OUT / "run-summary.json").write_text(
        json.dumps(
            {"control": c, "treatment": t, "candidate": "T-final at update 108 only"},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    publish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "execute", "publish"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "execute":
        execute()
    else:
        publish()


if __name__ == "__main__":
    main()
