"""Independent read-only replay and verification of seed542 fixed-update C/D."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train, verify_seed542_amended_corpus

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(model: train.PolicyValueNet) -> str:
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(name.encode() + b"\0")
        value.update(str(array.dtype).encode() + b"\0")
        value.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        value.update(array.tobytes())
    return value.hexdigest()


def authenticate(
    root: Path, output: Path, tablebase: Path | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    amendment = json.loads(
        (
            root / "docs/data/seed542-selfplay-4096/continuation-amendment.json"
        ).read_text()
    )
    protocol_path = root / "docs/data/seed542-selfplay-4096/training-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    binding = json.loads((output / "execution-binding.json").read_text())
    registration_path = output / "registration.json"
    registration = json.loads(registration_path.read_text())
    verify_seed542_amended_corpus.verify(root=root, tablebase_path=tablebase)
    if binding["registration_sha256"] != digest(registration_path):
        raise ValueError("fixed_update_registration_binding_mismatch")
    if registration.get("status") != "prospective frozen before training":
        raise ValueError("fixed_update_registration_not_prospective")
    for relative, expected in registration["input_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"fixed_update_registered_input_hash_mismatch:{relative}")
    for relative, expected in registration["execution_source_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(
                f"fixed_update_registered_execution_source_mismatch:{relative}"
            )
    if binding["protocol_sha256"] != digest(protocol_path):
        raise ValueError("fixed_update_protocol_binding_mismatch")
    if binding["initializer_sha256"] != digest(
        root / "docs/data/seed542-selfplay-4096/initializer.npz"
    ):
        raise ValueError("fixed_update_initializer_binding_mismatch")
    if binding["train_source_sha256"] != digest(root / "ml/alphazero_lite/train.py"):
        raise ValueError("fixed_update_train_source_binding_mismatch")
    if binding["runner_sha256"] != digest(
        root / "ml/alphazero_lite/run_seed542_fixed_update_cd.py"
    ):
        raise ValueError("fixed_update_runner_binding_mismatch")
    bound_paths = {
        "source_manifest_sha256": "docs/data/seed542-selfplay-4096/source-manifest.json",
        "amended_accounting_sha256": "docs/data/seed542-selfplay-4096/amended-accounting.json",
        "development_membership_sha256": "docs/data/seed542-selfplay-4096/development-membership.json",
        "seed504_manifest_sha256": "docs/data/seed504/manifest.json",
        "seed504_permutations_sha256": "docs/data/seed504/permutations.json",
    }
    for name, relative in bound_paths.items():
        if binding[name] != digest(root / relative):
            raise ValueError(f"fixed_update_input_binding_mismatch:{name}")
    for epoch in range(1, 5):
        if binding["seed504_checkpoints"][str(epoch)] != digest(
            root / f"docs/data/seed504/seed504_epoch{epoch}.npz"
        ):
            raise ValueError(
                f"fixed_update_historical_checkpoint_binding_mismatch:E{epoch}"
            )
    if binding["python"] != sys.version or binding["platform"] != platform.platform():
        raise ValueError("fixed_update_runtime_binding_mismatch")
    if binding["numpy"] != np.__version__ or binding["torch"] != torch.__version__:
        raise ValueError("fixed_update_runtime_package_binding_mismatch")
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
        raise ValueError("fixed_update_complete_package_binding_mismatch")
    if binding["verifier_sha256"] != digest(
        root / "ml/alphazero_lite/verify_seed542_fixed_update_cd.py"
    ):
        raise ValueError("fixed_update_verifier_binding_mismatch")
    if binding["test_sha256"] != digest(
        root / "ml/alphazero_lite/test_seed542_fixed_update_cd.py"
    ):
        raise ValueError("fixed_update_test_binding_mismatch")
    merged = protocol["updates"]
    control = protocol["control"]["schedule"]
    expanded = protocol["expanded"]["schedule"]
    if not (len(merged) == len(control) == len(expanded) == 108):
        raise ValueError("fixed_update_protocol_length_mismatch")
    normalized = []
    for index, (old, c_item, d_item) in enumerate(
        zip(merged, control, expanded, strict=True), 1
    ):
        expected_length = 412 if index % 27 == 0 else 512
        if (
            c_item["update"] != index
            or d_item["update"] != index
            or c_item["batch_size"] != expected_length
            or d_item["batch_size"] != expected_length
            or old["batch_size"] != expected_length
            or old["C"] != c_item["rows"]
            or old["D"] != d_item["rows"]
        ):
            raise ValueError(f"fixed_update_published_schedule_mismatch:{index}")
        normalized.append(
            {
                "update": index,
                "batch_size": expected_length,
                "C": c_item["rows"],
                "D": d_item["rows"],
            }
        )
    return amendment, normalized


def replay(arm: str, root: Path, updates: list[dict[str, Any]]) -> None:
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
    identities = []
    for item, path in zip(sources, paths, strict=True):
        if digest(path) != item["sha256"]:
            raise ValueError(f"fixed_update_source_hash_mismatch:{item['path']}")
        identities.extend(
            f"{item['path']}:{line}"
            for line, _ in enumerate(path.read_bytes().splitlines(), 1)
        )
    x, p, v, _replay, policy_weights = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    positions = {row_id: i for i, row_id in enumerate(identities)}
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        model, root / "docs/data/seed542-selfplay-4096/initializer.npz"
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    output = root / "docs/data/seed542-fixed-update-cd"
    trace_path = output / f"{arm}-updates.jsonl"
    recorded = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if len(recorded) != 108:
        raise ValueError(f"fixed_update_trace_count_mismatch:{arm}")
    for schedule_item, trace_item in zip(updates, recorded, strict=True):
        batch = schedule_item[arm]
        if (
            trace_item["update"] != schedule_item["update"]
            or trace_item["row_ids"] != batch
        ):
            raise ValueError(
                f"fixed_update_order_mismatch:{arm}:{schedule_item['update']}"
            )
        index = np.asarray([positions[row_id] for row_id in batch], dtype=np.int64)
        bx, bp, bv = (
            torch.from_numpy(x[index]),
            torch.from_numpy(p[index]),
            torch.from_numpy(v[index]),
        )
        bw = torch.from_numpy(policy_weights[index])
        mask = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[index]))
        logits, value = model(bx)
        policy = train.weighted_policy_loss(
            train.compute_policy_cross_entropy(logits.masked_fill(mask <= 0, -1e9), bp),
            bw,
        )
        value_loss = train.compute_value_loss_vector(
            value, bv, value_loss="huber", huber_delta=1.0
        ).mean()
        total = policy + 0.5 * value_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        actual = {
            "policy_loss": float(policy.detach()),
            "value_loss": float(value_loss.detach()),
            "weighted_total_objective": float(total.detach()),
            "gradient_norm_preclip": norm,
            "optimizer_step": int(
                optimizer.state[next(model.parameters())]["step"].item()
            ),
            "parameter_state_sha256": state_hash(model),
        }
        for field, expected in actual.items():
            observed = trace_item[field]
            if isinstance(expected, float):
                if observed != expected:
                    raise ValueError(
                        f"fixed_update_record_mismatch:{arm}:{schedule_item['update']}:{field}"
                    )
            elif observed != expected:
                raise ValueError(
                    f"fixed_update_record_mismatch:{arm}:{schedule_item['update']}:{field}"
                )
        if schedule_item["update"] in {27, 54, 81, 108}:
            checkpoint_path = output / f"{arm}-update-{schedule_item['update']:03d}.npz"
            with np.load(checkpoint_path) as saved:
                expected_state = train.checkpoint_from_model(model)
                if set(saved.files) != set(expected_state):
                    raise ValueError(
                        f"fixed_update_checkpoint_names_mismatch:{arm}:{schedule_item['update']}"
                    )
                for name, array in expected_state.items():
                    if not np.array_equal(saved[name], array):
                        raise ValueError(
                            f"fixed_update_checkpoint_array_mismatch:{arm}:{schedule_item['update']}:{name}"
                        )
            if arm == "C":
                historical = (
                    root
                    / f"docs/data/seed504/seed504_epoch{schedule_item['update'] // 27}.npz"
                )
                with np.load(historical) as old:
                    if set(old.files) != set(expected_state) or any(
                        not np.array_equal(old[name], expected_state[name])
                        for name in expected_state
                    ):
                        raise ValueError(
                            f"seed504_control_reproduction_mismatch:E{schedule_item['update'] // 27}"
                        )


def verify_development(root: Path, output: Path) -> None:
    corpus = root / "docs/data/seed542-selfplay-4096"
    manifest = json.loads((corpus / "source-manifest.json").read_text())
    development_sources = [
        source for source in manifest["sources"] if source["role"] == "development"
    ]
    paths = [root / source["path"] for source in development_sources]
    row_ids = []
    raw_rows = []
    for source, path in zip(development_sources, paths, strict=True):
        if digest(path) != source["sha256"]:
            raise ValueError(
                f"fixed_update_development_source_hash_mismatch:{source['path']}"
            )
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            row_ids.append(f"{source['path']}:{line_number}")
            raw_rows.append(json.loads(line))
    membership = json.loads((corpus / "development-membership.json").read_text())
    retained = {item["row_id"] for item in membership["membership"] if item["included"]}
    x_all, p_all, v_all, _replay_indexes, _policy_weights = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    selected = [i for i, row_id in enumerate(row_ids) if row_id in retained]
    if len(selected) != len(retained):
        raise ValueError("fixed_update_retained_membership_mismatch")
    x, p, v = x_all[selected], p_all[selected], v_all[selected, 0]
    selected_ids = [row_ids[index] for index in selected]
    buckets = (">32", "17-32", "<=16")
    endpoints = {
        "initializer": root / "docs/data/seed542-selfplay-4096/initializer.npz",
        "C-final": output / "C-update-108.npz",
        "D-final": output / "D-update-108.npz",
    }
    recomputed: dict[str, Any] = {}
    for endpoint, checkpoint in endpoints.items():
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(model, checkpoint)
        model.eval()
        with torch.no_grad():
            raw_logits, raw_values = model(torch.from_numpy(x))
        legal = train.legal_mask_matrix_for_encoded_states(x).astype(bool)
        logits = raw_logits.numpy().copy()
        logits[~legal] = -1e9
        log_probabilities = logits - np.logaddexp.reduce(logits, axis=1)[:, None]
        probabilities = np.exp(log_probabilities)
        policy_losses = -(p * log_probabilities).sum(axis=1)
        value_predictions = raw_values.numpy().reshape(-1)
        value_losses = (value_predictions - v) ** 2
        cohort_indices: dict[str, list[int]] = {
            "pooled": list(range(len(selected_ids))),
            **{name: [] for name in buckets},
        }
        for index in range(len(selected_ids)):
            stones = int(round(float(np.sum(x[index, :12])) * 48.0))
            name = ">32" if stones > 32 else "17-32" if stones >= 17 else "<=16"
            cohort_indices[name].append(index)
        metrics = {}
        for cohort, indices in cohort_indices.items():
            identities: dict[bytes, list[int]] = {}
            for index in indices:
                identities.setdefault(x[index].tobytes(), []).append(index)
            metrics[cohort] = {
                "rows": len(indices),
                "identities": len(identities),
                "exposure_weighted": (
                    {
                        "policy_ce": float(policy_losses[indices].mean()),
                        "actual_value_mse": float(value_losses[indices].mean()),
                    }
                    if indices
                    else None
                ),
                "equal_production_float32_input_weighted": (
                    {
                        "policy_ce": float(
                            np.mean(
                                [policy_losses[ix].mean() for ix in identities.values()]
                            )
                        ),
                        "actual_value_mse": float(
                            np.mean(
                                [value_losses[ix].mean() for ix in identities.values()]
                            )
                        ),
                    }
                    if indices
                    else None
                ),
            }
        predictions = [
            {
                "row_id": row_id,
                "production_float32_input_sha256": hashlib.sha256(
                    x[index].tobytes()
                ).hexdigest(),
                "target_policy": p[index].tolist(),
                "target_value": float(v[index]),
                "legal_mask": legal[index].tolist(),
                "policy_logits_legal_masked": logits[index].tolist(),
                "policy_probabilities_legal_masked": probabilities[index].tolist(),
                "value_prediction": float(value_predictions[index]),
            }
            for index, row_id in enumerate(selected_ids)
        ]
        recomputed[endpoint] = {
            "checkpoint_sha256": digest(checkpoint),
            "cohort_row_count": len(selected_ids),
            "cohort_identity_count": len({row.tobytes() for row in x}),
            "metrics": metrics,
            "predictions": predictions,
        }
    effect_values = {}
    for comparison, left, right in (
        ("D_minus_C", "D-final", "C-final"),
        ("D_minus_initializer", "D-final", "initializer"),
    ):
        effect_values[comparison] = {}
        for cohort in ("pooled", *buckets):
            effect_values[comparison][cohort] = {}
            for weighting in (
                "exposure_weighted",
                "equal_production_float32_input_weighted",
            ):
                effect_values[comparison][cohort][weighting] = {}
                left_values = recomputed[left]["metrics"][cohort][weighting]
                right_values = recomputed[right]["metrics"][cohort][weighting]
                for metric in ("policy_ce", "actual_value_mse"):
                    effect_values[comparison][cohort][weighting][metric] = (
                        left_values[metric] - right_values[metric]
                        if left_values is not None and right_values is not None
                        else None
                    )
    expected = {
        "development_membership_sha256": digest(corpus / "development-membership.json"),
        "interpretation": "Descriptive development diagnostics only; D-final is not advanced by these metrics. Intended later composition is D-final policy/trunk with the frozen initializer value/trunk.",
        "effects": effect_values,
        "endpoints": recomputed,
    }
    actual = json.loads((output / "development-evaluation.json").read_text())
    if json.dumps(expected, sort_keys=True, separators=(",", ":")) != json.dumps(
        actual, sort_keys=True, separators=(",", ":")
    ):
        raise ValueError(
            "fixed_update_development_predictions_metrics_or_effects_mismatch"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs/data/seed542-fixed-update-cd"
    )
    parser.add_argument(
        "--tablebase",
        type=Path,
        default=Path("/home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb"),
    )
    args = parser.parse_args()
    root, output = args.root.resolve(), args.out.resolve()
    _amendment, schedule = authenticate(root, output, args.tablebase)
    replay("C", root, schedule)
    replay("D", root, schedule)
    verify_development(root, output)
    print(
        json.dumps(
            {"status": "training_replay_verified", "arms": ["C", "D"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
