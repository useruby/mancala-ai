"""Run the frozen seed542 fixed-update C/D training comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
PROTOCOL = CORPUS / "training-protocol.json"
INITIALIZER = CORPUS / "initializer.npz"
OUT = ROOT / "docs/data/seed542-fixed-update-cd"
BOUNDARIES = {27, 54, 81, 108}
BUCKETS = (">32", "17-32", "<=16")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_sources(
    arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if arm == "C":
        manifest = json.loads((ROOT / "docs/data/seed504/manifest.json").read_text())
        sources = manifest["sources"]["training"]
    elif arm == "D":
        manifest = json.loads((CORPUS / "source-manifest.json").read_text())
        sources = [s for s in manifest["sources"] if s["role"] == "training"]
    else:
        raise ValueError(f"unknown arm: {arm}")
    paths = [ROOT / source["path"] for source in sources]
    rows: list[str] = []
    for source, path in zip(sources, paths, strict=True):
        if sha(path) != source["sha256"]:
            raise ValueError(f"training_source_sha256_mismatch:{source['path']}")
        rows.extend(
            f"{source['path']}:{line_number}"
            for line_number, _line in enumerate(path.read_bytes().splitlines(), 1)
        )
    x, p, v, _replay, weights = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    if len(x) != len(rows):
        raise ValueError(f"loader_row_identity_count_mismatch:{arm}")
    return x, p, v, weights, rows


def load_schedule() -> dict[str, Any]:
    schedule = json.loads(PROTOCOL.read_text())
    if schedule["protocol"]["status"] != "frozen prospective; NOT EXECUTED":
        raise ValueError("published_protocol_not_prospective")
    merged = schedule["updates"]
    control_schedule = schedule["control"]["schedule"]
    expanded_schedule = schedule["expanded"]["schedule"]
    if (
        len(merged) != 108
        or len(control_schedule) != 108
        or len(expanded_schedule) != 108
        or [u["batch_size"] for u in merged]
        != [n for _ in range(4) for n in ([512] * 26 + [412])]
        or [u["update"] for u in control_schedule] != list(range(1, 109))
        or [u["update"] for u in expanded_schedule] != list(range(1, 109))
    ):
        raise ValueError("published_schedule_shape_mismatch")
    normalized = []
    for index, (control, expanded, combined) in enumerate(
        zip(control_schedule, expanded_schedule, merged, strict=True), 1
    ):
        if (
            control["batch_size"] != combined["batch_size"]
            or expanded["batch_size"] != combined["batch_size"]
            or control["rows"] != combined["C"]
            or expanded["rows"] != combined["D"]
        ):
            raise ValueError(f"published_schedule_views_disagree:{index}")
        normalized.append(
            {
                "update": index,
                "batch_size": combined["batch_size"],
                "C": control["rows"],
                "D": expanded["rows"],
            }
        )
    old_manifest = json.loads((ROOT / "docs/data/seed504/manifest.json").read_text())
    old_permutation_path = ROOT / "docs/data/seed504/permutations.json"
    if sha(old_permutation_path) != old_manifest["permutations_sha256"]:
        raise ValueError("seed504_frozen_permutations_sha256_mismatch")
    old_permutations = json.loads(old_permutation_path.read_text())
    frozen_control_rows = [
        f"{source['path']}:{line}"
        for source in old_manifest["sources"]["training"]
        for line, _row in enumerate(
            (ROOT / source["path"]).read_bytes().splitlines(), 1
        )
    ]
    if len(old_permutations) != 4 or any(
        len(order) != len(frozen_control_rows) for order in old_permutations
    ):
        raise ValueError("seed504_frozen_permutation_shape_mismatch")
    expected_control = [
        frozen_control_rows[index] for order in old_permutations for index in order
    ]
    actual_control = [row for item in normalized for row in item["C"]]
    expanded_rows = [row for item in normalized for row in item["D"]]
    if actual_control != expected_control:
        raise ValueError("published_control_schedule_differs_from_seed504_permutations")
    if len(expanded_rows) != 54896 or len(set(expanded_rows)) != 54896:
        raise ValueError("published_expanded_schedule_coverage_mismatch")
    schedule["updates"] = normalized
    return schedule


def load_model() -> train.PolicyValueNet:
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, INITIALIZER)
    return model.cpu()


def state_digest(model: train.PolicyValueNet) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b"\0")
        h.update(str(tensor.dtype).encode() + b"\0")
        h.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        h.update(tensor.tobytes())
    return h.hexdigest()


def parameter_checkpoint(model: train.PolicyValueNet, path: Path) -> None:
    np.savez(path, **train.checkpoint_from_model(model))


def update(
    model: train.PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch_ids: list[str],
    row_positions: dict[str, int],
    x: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    policy_weights: np.ndarray,
) -> dict[str, Any]:
    indices = np.asarray([row_positions[row] for row in batch_ids], dtype=np.int64)
    bx = torch.from_numpy(x[indices])
    bp = torch.from_numpy(p[indices])
    bv = torch.from_numpy(v[indices])
    bw = torch.from_numpy(policy_weights[indices])
    legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[indices]))
    model.train()
    logits, value_prediction = model(bx)
    policy_loss = train.weighted_policy_loss(
        train.compute_policy_cross_entropy(logits.masked_fill(legal <= 0.0, -1e9), bp),
        bw,
    )
    value_loss = train.compute_value_loss_vector(
        value_prediction, bv, value_loss="huber", huber_delta=1.0
    ).mean()
    objective = policy_loss + 0.5 * value_loss
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    preclip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    return {
        "batch_size": len(batch_ids),
        "row_ids": batch_ids,
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "weighted_total_objective": float(objective.detach()),
        "gradient_norm_preclip": preclip_norm,
        "optimizer_step": int(optimizer.state[next(model.parameters())]["step"].item()),
        "parameter_state_sha256": state_digest(model),
    }


def verify_seed504_checkpoint(
    model: train.PolicyValueNet, epoch: int
) -> dict[str, Any]:
    historical_path = ROOT / f"docs/data/seed504/seed504_epoch{epoch}.npz"
    current = train.checkpoint_from_model(model)
    with np.load(historical_path) as historical:
        if set(current) != set(historical.files):
            raise ValueError(f"seed504_E{epoch}_parameter_names_mismatch")
        for name in current:
            expected = historical[name]
            actual = current[name]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError(f"seed504_E{epoch}_parameter_metadata_mismatch:{name}")
            if not np.array_equal(actual, expected):
                mismatch = int(np.count_nonzero(actual != expected))
                raise ValueError(
                    f"seed504_E{epoch}_parameter_array_mismatch:{name}:elements={mismatch}"
                )
    return {"checkpoint": str(historical_path.relative_to(ROOT)), "bitwise": True}


def development_rows() -> list[dict[str, Any]]:
    manifest = json.loads((CORPUS / "source-manifest.json").read_text())
    sources = [
        source for source in manifest["sources"] if source["role"] == "development"
    ]
    membership = json.loads((CORPUS / "development-membership.json").read_text())
    retained = {
        row["row_id"] for row in membership["membership"] if row["included"] is True
    }
    raw_rows = []
    identities = []
    for source in sources:
        path = ROOT / source["path"]
        if sha(path) != source["sha256"]:
            raise ValueError(f"development_source_sha256_mismatch:{source['path']}")
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            identity = f"{source['path']}:{line_number}"
            row = json.loads(line)
            row["_row_id"] = identity
            raw_rows.append(row)
            identities.append(identity)
    dev_paths = [ROOT / source["path"] for source in sources]
    x_all, p_all, v_all, _replay, _weights = train.load_jsonl_replay(
        dev_paths,
        weights=[1] * len(dev_paths),
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    rows = []
    for index, identity in enumerate(identities):
        if identity in retained:
            raw_rows[index]["state"] = x_all[index].tolist()
            raw_rows[index]["policy"] = p_all[index].tolist()
            raw_rows[index]["value"] = float(v_all[index, 0])
            rows.append(raw_rows[index])
    if {row["_row_id"] for row in rows} != retained:
        raise ValueError("retained_development_membership_reconstruction_mismatch")
    return rows


def evaluate_checkpoint(
    checkpoint: Path, rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, checkpoint)
    model.eval()
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    p = np.asarray([row["policy"] for row in rows], dtype=np.float32)
    v = np.asarray([row["value"] for row in rows], dtype=np.float32)
    legal = train.legal_mask_matrix_for_encoded_states(x).astype(bool)
    with torch.no_grad():
        logits_tensor, values_tensor = model(torch.from_numpy(x))
    logits = logits_tensor.numpy().copy()
    logits[~legal] = -1e9
    log_probs = logits - np.logaddexp.reduce(logits, axis=1)[:, None]
    probabilities = np.exp(log_probs)
    policy_ce = -(p * log_probs).sum(axis=1)
    value_mse = (values_tensor.numpy().reshape(-1) - v) ** 2
    groups: dict[str, list[int]] = {"pooled": list(range(len(rows)))}
    groups.update({name: [] for name in BUCKETS})
    for index, _row in enumerate(rows):
        active = int(round(float(np.sum(x[index, :12])) * 48.0))
        bucket = ">32" if active > 32 else "17-32" if active >= 17 else "<=16"
        groups[bucket].append(index)
    metrics: dict[str, Any] = {}
    for name, indices in groups.items():
        identities: dict[bytes, list[int]] = defaultdict(list)
        for index in indices:
            identities[x[index].tobytes()].append(index)
        metrics[name] = {
            "rows": len(indices),
            "identities": len(identities),
            "exposure_weighted": (
                {
                    "policy_ce": float(policy_ce[indices].mean()),
                    "actual_value_mse": float(value_mse[indices].mean()),
                }
                if indices
                else None
            ),
            "equal_production_float32_input_weighted": (
                {
                    "policy_ce": float(
                        np.mean([policy_ce[ix].mean() for ix in identities.values()])
                    ),
                    "actual_value_mse": float(
                        np.mean([value_mse[ix].mean() for ix in identities.values()])
                    ),
                }
                if indices
                else None
            ),
        }
    predictions = [
        {
            "row_id": row["_row_id"],
            "production_float32_input_sha256": hashlib.sha256(
                x[index].tobytes()
            ).hexdigest(),
            "target_policy": p[index].tolist(),
            "target_value": float(v[index]),
            "legal_mask": legal[index].tolist(),
            "policy_logits_legal_masked": logits[index].tolist(),
            "policy_probabilities_legal_masked": probabilities[index].tolist(),
            "value_prediction": float(values_tensor[index].item()),
        }
        for index, row in enumerate(rows)
    ]
    return metrics, predictions


def evaluate_endpoints(out: Path) -> dict[str, Any]:
    rows = development_rows()
    endpoints = {
        "initializer": INITIALIZER,
        "C-final": out / "C-update-108.npz",
        "D-final": out / "D-update-108.npz",
    }
    evaluations = {}
    for name, checkpoint in endpoints.items():
        metrics, predictions = evaluate_checkpoint(checkpoint, rows)
        evaluations[name] = {
            "checkpoint_sha256": sha(checkpoint),
            "cohort_row_count": len(rows),
            "cohort_identity_count": len(
                {np.asarray(row["state"], dtype=np.float32).tobytes() for row in rows}
            ),
            "metrics": metrics,
            "predictions": predictions,
        }
    effects = {}
    for comparison, left, right in (
        ("D_minus_C", "D-final", "C-final"),
        ("D_minus_initializer", "D-final", "initializer"),
    ):
        effects[comparison] = {}
        for cohort in ("pooled", *BUCKETS):
            effects[comparison][cohort] = {}
            for weighting in (
                "exposure_weighted",
                "equal_production_float32_input_weighted",
            ):
                effects[comparison][cohort][weighting] = {}
                left_metrics = evaluations[left]["metrics"][cohort][weighting]
                right_metrics = evaluations[right]["metrics"][cohort][weighting]
                for metric in ("policy_ce", "actual_value_mse"):
                    effects[comparison][cohort][weighting][metric] = (
                        left_metrics[metric] - right_metrics[metric]
                        if left_metrics is not None and right_metrics is not None
                        else None
                    )
    result = {
        "development_membership_sha256": sha(CORPUS / "development-membership.json"),
        "interpretation": "Descriptive development diagnostics only; D-final is not advanced by these metrics. Intended later composition is D-final policy/trunk with the frozen initializer value/trunk.",
        "effects": effects,
        "endpoints": evaluations,
    }
    (out / "development-evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def freeze_registration(out: Path) -> dict[str, Any]:
    old_manifest = json.loads((ROOT / "docs/data/seed504/manifest.json").read_text())
    expanded_manifest = json.loads((CORPUS / "source-manifest.json").read_text())
    input_paths = {
        str(INITIALIZER.relative_to(ROOT)),
        str(PROTOCOL.relative_to(ROOT)),
        "docs/data/seed542-selfplay-4096/continuation-amendment.json",
        "docs/data/seed542-selfplay-4096/amended-accounting.json",
        "docs/data/seed542-selfplay-4096/source-manifest.json",
        "docs/data/seed542-selfplay-4096/development-membership.json",
        "docs/data/seed542-selfplay-4096/recovered-seed542001-complete.jsonl",
        "docs/data/seed542-selfplay-4096/recovered-seed542001-provenance.jsonl",
        "docs/data/seed504/manifest.json",
        "docs/data/seed504/permutations.json",
        *(
            source["path"]
            for source in [
                *old_manifest["sources"]["training"],
                *old_manifest["sources"]["development"],
                *expanded_manifest["sources"],
            ]
        ),
        *(f"docs/data/seed504/seed504_epoch{epoch}.npz" for epoch in range(1, 5)),
    }
    source_paths = {
        relative: sha(ROOT / relative)
        for relative in sorted(input_paths)
        if (ROOT / relative).is_file()
    }
    source_code = {
        relative: sha(ROOT / relative)
        for relative in (
            "ml/alphazero_lite/train.py",
            "ml/alphazero_lite/run_seed542_fixed_update_cd.py",
            "ml/alphazero_lite/verify_seed542_fixed_update_cd.py",
            "ml/alphazero_lite/test_seed542_fixed_update_cd.py",
        )
    }
    amendment = json.loads((CORPUS / "continuation-amendment.json").read_text())
    packages = sorted(
        (
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name") and dist.metadata["Name"].lower() != "pip"
        ),
        key=str.casefold,
    )
    if packages != amendment["interpreter"]["installed_packages"]:
        raise ValueError("registered_training_environment_package_set_mismatch")
    registration = {
        "status": "prospective frozen before training",
        "comparison": "seed542 fixed-update C (original seed504 data) versus D (expanded seed542 data)",
        "protocol_sha256": sha(PROTOCOL),
        "input_sha256": source_paths,
        "execution_source_sha256": source_code,
        "runtime": {
            "interpreter": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "installed_packages": packages,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "planned_commands": [
            f"prlimit --as=8589934592 -- nice -n 19 taskset -c 31 {sys.executable} -m ml.alphazero_lite.run_seed542_fixed_update_cd --arm both --out {out}",
            f"{sys.executable} -m ml.alphazero_lite.verify_seed542_fixed_update_cd --out {out} --tablebase /home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb",
        ],
        "protocol": {
            "updates_per_arm": 108,
            "batch_sizes": [n for _ in range(4) for n in ([512] * 26 + [412])],
            "exposures_per_arm": 54896,
            "policy_objective": "production legal-masked weighted policy CE with exact-root coefficient 1",
            "value_objective": "production Huber(value, delta=1)",
            "value_coefficient": 0.5,
            "optimizer": "fresh Adam per arm; lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0",
            "gradient_clip_norm": 1.0,
            "scheduler": "none",
            "candidate_endpoint": "update 108 only",
        },
    }
    path = out / "registration.json"
    if path.exists():
        raise FileExistsError("prospective_registration_already_exists")
    path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    return registration


def run_arm(arm: str, out: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    x, p, v, policy_weights, identities = load_sources(arm)
    row_positions = {identity: position for position, identity in enumerate(identities)}
    model = load_model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    records = []
    control_checks = []
    record_path = out / f"{arm}-updates.jsonl"
    with record_path.open("w", encoding="utf-8") as trace:
        for item in schedule["updates"]:
            row_ids = item[arm]
            record = update(
                model, optimizer, row_ids, row_positions, x, p, v, policy_weights
            )
            record["update"] = item["update"]
            records.append(record)
            trace.write(json.dumps(record, sort_keys=True) + "\n")
            trace.flush()
            if item["update"] in BOUNDARIES:
                boundary = item["update"]
                checkpoint = out / f"{arm}-update-{boundary:03d}.npz"
                parameter_checkpoint(model, checkpoint)
                if arm == "C":
                    control_checks.append(
                        verify_seed504_checkpoint(model, boundary // 27)
                    )
    return {
        "updates": len(records),
        "exposures": sum(record["batch_size"] for record in records),
        "trace_sha256": sha(record_path),
        "checkpoints": {
            str(n): sha(out / f"{arm}-update-{n:03d}.npz") for n in sorted(BOUNDARIES)
        },
        "control_reproduction": control_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--arm", choices=("C", "D", "both"), default="both")
    args = parser.parse_args()
    if (
        sha(INITIALIZER)
        != "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
    ):
        raise ValueError("initializer_sha256_mismatch")
    schedule = load_schedule()
    args.out.mkdir(parents=True, exist_ok=True)
    freeze_registration(args.out)
    source_binding = {
        "status": "prospective frozen before training",
        "registration_sha256": sha(args.out / "registration.json"),
        "protocol_sha256": sha(PROTOCOL),
        "initializer_sha256": sha(INITIALIZER),
        "source_manifest_sha256": sha(CORPUS / "source-manifest.json"),
        "amended_accounting_sha256": sha(CORPUS / "amended-accounting.json"),
        "development_membership_sha256": sha(CORPUS / "development-membership.json"),
        "seed504_manifest_sha256": sha(ROOT / "docs/data/seed504/manifest.json"),
        "seed504_permutations_sha256": sha(
            ROOT / "docs/data/seed504/permutations.json"
        ),
        "seed504_checkpoints": {
            str(epoch): sha(ROOT / f"docs/data/seed504/seed504_epoch{epoch}.npz")
            for epoch in range(1, 5)
        },
        "train_source_sha256": sha(Path(train.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "verifier_sha256": sha(
            ROOT / "ml/alphazero_lite/verify_seed542_fixed_update_cd.py"
        ),
        "test_sha256": sha(ROOT / "ml/alphazero_lite/test_seed542_fixed_update_cd.py"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "device": "cpu",
        "command": sys.argv,
        "outer_command": os.environ.get("SEED542_CD_LAUNCH_COMMAND"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.out / "execution-binding.json").write_text(
        json.dumps(source_binding, indent=2, sort_keys=True) + "\n"
    )
    if args.arm in {"C", "both"}:
        control = run_arm("C", args.out, schedule)
        (args.out / "C-control-reproduction.json").write_text(
            json.dumps(control, indent=2, sort_keys=True) + "\n"
        )
        if not all(check["bitwise"] for check in control["control_reproduction"]):
            return
    if args.arm in {"D", "both"}:
        if args.arm == "D":
            prior = json.loads((args.out / "C-control-reproduction.json").read_text())
            if not prior["control_reproduction"] or not all(
                item["bitwise"] for item in prior["control_reproduction"]
            ):
                raise ValueError("control_reproduction_not_confirmed")
        expanded = run_arm("D", args.out, schedule)
        (args.out / "D-training-summary.json").write_text(
            json.dumps(expanded, indent=2, sort_keys=True) + "\n"
        )
        if args.arm == "both":
            evaluate_endpoints(args.out)
        elif args.arm == "D":
            evaluate_endpoints(args.out)


if __name__ == "__main__":
    main()
