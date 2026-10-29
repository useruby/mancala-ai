"""Publish development predictions and descriptive effects for seed542 C/D v2."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
BUCKETS = (">32", "17-32", "<=16")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retained_development() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((CORPUS / "source-manifest.json").read_text())
    sources = [row for row in manifest["sources"] if row["role"] == "development"]
    membership = json.loads((CORPUS / "development-membership.json").read_text())
    retained = {row["row_id"] for row in membership["membership"] if row["included"]}
    paths = [ROOT / source["path"] for source in sources]
    identities = []
    for source, path in zip(sources, paths, strict=True):
        if sha(path) != source["sha256"]:
            raise ValueError(f"v2_development_source_sha256_mismatch:{source['path']}")
        identities.extend(
            f"{source['path']}:{line}"
            for line, _content in enumerate(path.read_bytes().splitlines(), 1)
        )
    x, p, v, _replay, _weights = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
        include_policy_loss_weights=True,
    )
    if len(identities) != len(x) or set(identities) != {
        row["row_id"] for row in membership["membership"]
    }:
        raise ValueError("v2_development_membership_source_mismatch")
    indices = [index for index, row_id in enumerate(identities) if row_id in retained]
    return (
        [identities[index] for index in indices],
        x[indices],
        p[indices],
        v[indices, 0],
    )


def evaluate(
    checkpoint: Path,
    names: list[str],
    x: np.ndarray,
    p: np.ndarray,
    target_value: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    predicted_value = value_tensor.numpy().reshape(-1)
    policy_loss = -(p * log_probs).sum(axis=1)
    value_loss = (predicted_value - target_value) ** 2
    cohorts: dict[str, list[int]] = {"pooled": list(range(len(names)))}
    cohorts.update({name: [] for name in BUCKETS})
    for index in range(len(names)):
        stones = int(round(float(np.sum(x[index, :12])) * 48.0))
        bucket = ">32" if stones > 32 else "17-32" if stones >= 17 else "<=16"
        cohorts[bucket].append(index)
    metrics = {}
    for cohort, indices in cohorts.items():
        by_input: dict[bytes, list[int]] = defaultdict(list)
        for index in indices:
            by_input[x[index].tobytes()].append(index)
        metrics[cohort] = {
            "rows": len(indices),
            "production_float32_input_identities": len(by_input),
            "exposure_weighted": (
                {
                    "policy_ce": float(policy_loss[indices].mean()),
                    "actual_value_mse": float(value_loss[indices].mean()),
                }
                if indices
                else None
            ),
            "equal_production_float32_input_weighted": (
                {
                    "policy_ce": float(
                        np.mean(
                            [policy_loss[group].mean() for group in by_input.values()]
                        )
                    ),
                    "actual_value_mse": float(
                        np.mean(
                            [value_loss[group].mean() for group in by_input.values()]
                        )
                    ),
                }
                if indices
                else None
            ),
        }
    rows = [
        {
            "endpoint": checkpoint.stem,
            "row_id": names[index],
            "production_float32_input_sha256": hashlib.sha256(
                x[index].tobytes()
            ).hexdigest(),
            "target_policy": p[index].tolist(),
            "target_actual_value": float(target_value[index]),
            "legal_mask": legal[index].tolist(),
            "policy_logits_legal_masked": logits[index].tolist(),
            "policy_probabilities_legal_masked": probabilities[index].tolist(),
            "value_prediction": float(predicted_value[index]),
        }
        for index in range(len(names))
    ]
    return metrics, rows


def publish(output: Path) -> dict[str, Any]:
    names, x, p, value = retained_development()
    endpoints = {
        "initializer": CORPUS / "initializer.npz",
        "C-final": output / "C-update-108.npz",
        "D-final": output / "D-update-108.npz",
    }
    endpoint_results = {}
    prediction_rows = []
    for name, checkpoint in endpoints.items():
        metrics, rows = evaluate(checkpoint, names, x, p, value)
        endpoint_results[name] = {
            "checkpoint_sha256": sha(checkpoint),
            "cohort_row_count": len(names),
            "cohort_identity_count": len({state.tobytes() for state in x}),
            "metrics": metrics,
        }
        for row in rows:
            row["endpoint"] = name
            prediction_rows.append(row)
    effects = {}
    for effect_name, left, right in (
        ("D_minus_C", "D-final", "C-final"),
        ("D_minus_initializer", "D-final", "initializer"),
    ):
        effects[effect_name] = {}
        for cohort in ("pooled", *BUCKETS):
            effects[effect_name][cohort] = {}
            for weighting in (
                "exposure_weighted",
                "equal_production_float32_input_weighted",
            ):
                a = endpoint_results[left]["metrics"][cohort][weighting]
                b = endpoint_results[right]["metrics"][cohort][weighting]
                effects[effect_name][cohort][weighting] = {
                    metric: a[metric] - b[metric]
                    if a is not None and b is not None
                    else None
                    for metric in ("policy_ce", "actual_value_mse")
                }
    with (output / "development-predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in prediction_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "development_membership_sha256": sha(CORPUS / "development-membership.json"),
        "sign_convention": "effect = first endpoint loss minus second endpoint loss; negative means improvement",
        "cohort_counts": {
            name: {
                "rows": values["rows"],
                "production_float32_input_identities": values[
                    "production_float32_input_identities"
                ],
            }
            for name, values in endpoint_results["initializer"]["metrics"].items()
        },
        "metrics": {name: value["metrics"] for name, value in endpoint_results.items()},
        "effects": effects,
        "endpoints": {
            name: {
                key: value[key]
                for key in (
                    "checkpoint_sha256",
                    "cohort_row_count",
                    "cohort_identity_count",
                )
            }
            for name, value in endpoint_results.items()
        },
        "interpretation": "Descriptive diagnostics only; no advancement threshold. Intended later runtime composition is D-final policy/trunk plus frozen initializer value/trunk; trained value metrics are diagnostic only.",
    }
    (output / "development-metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
