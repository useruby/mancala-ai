"""Independent endpoint metric and prediction recomputation for seed504."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ml.alphazero_lite import run_seed504, train


def recompute(checkpoint: Path, rows: list[dict]) -> tuple[dict, list[dict]]:
    model = run_seed504.load_model(checkpoint, torch.device("cpu"))
    inputs = np.asarray([row["state"] for row in rows], dtype=np.float32)
    targets = np.asarray([row["policy"] for row in rows], dtype=np.float32)
    values = np.asarray([row["value"] for row in rows], dtype=np.float32)
    legal = train.legal_mask_matrix_for_encoded_states(inputs).astype(bool)
    with torch.inference_mode():
        logits_tensor, predictions_tensor = model(torch.from_numpy(inputs))
    logits = logits_tensor.numpy().copy()
    logits[~legal] = -1e9
    maximum = logits.max(axis=1, keepdims=True)
    log_probabilities = (
        logits - maximum - np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))
    )
    probabilities = np.exp(log_probabilities)
    losses = -(targets * np.where(legal, log_probabilities, 0.0)).sum(axis=1)
    value_predictions = predictions_tensor.numpy().reshape(-1)
    squared_errors = np.square(value_predictions - values)
    identities: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        identities[run_seed504.keys(row)[1]].append(i)
    buckets: dict[str, np.ndarray] = {"pooled": np.arange(len(rows))}
    for name in (">32", "17-32", "<=16"):
        buckets[name] = np.asarray(
            [
                i
                for i, row in enumerate(rows)
                if run_seed504.bucket(run_seed504.stones(row)) == name
            ]
        )
    metrics = {}
    for name, indices in buckets.items():
        if not len(indices):
            metrics[name] = {"rows": 0, "exposure": None, "equal_exact_input": None}
            continue
        by_input: dict[tuple[float, ...], list[int]] = defaultdict(list)
        for i in indices:
            by_input[run_seed504.keys(rows[int(i)])[1]].append(int(i))
        metrics[name] = {
            "rows": len(indices),
            "exact_inputs": len(by_input),
            "exposure": {
                "policy_ce": float(losses[indices].mean()),
                "actual_value_mse": float(squared_errors[indices].mean()),
            },
            "equal_exact_input": {
                "policy_ce": float(
                    np.mean([losses[ix].mean() for ix in by_input.values()])
                ),
                "actual_value_mse": float(
                    np.mean([squared_errors[ix].mean() for ix in by_input.values()])
                ),
            },
        }
    evidence = [
        {
            "source": row["_source"],
            "line": row["_line"],
            "policy_logits": logits[i].tolist(),
            "policy_probabilities": probabilities[i].tolist(),
            "value_prediction": float(value_predictions[i]),
        }
        for i, row in enumerate(rows)
    ]
    return metrics, evidence


def main() -> None:
    out = Path("docs/data/seed504")
    _, development, _ = run_seed504.split()
    expected = json.loads((out / "results.json").read_text())
    for key, filename in (
        ("initializer", run_seed504.INITIALIZER),
        *[(f"E{i}", out / f"seed504_epoch{i}.npz") for i in range(1, 5)],
    ):
        metrics, predictions = recompute(filename, development)
        published_metrics = expected[key]["metrics"]
        for group, result in metrics.items():
            assert result["rows"] == published_metrics[group]["rows"]
            if result["rows"]:
                for weighting in ("exposure", "equal_exact_input"):
                    for metric, value in result[weighting].items():
                        assert np.isclose(
                            value,
                            published_metrics[group][weighting][metric],
                            rtol=0,
                            atol=1e-6,
                        ), (key, group, weighting, metric)
        published = expected[key]["predictions"]
        for actual, recorded in zip(predictions, published, strict=True):
            assert actual["source"] == recorded["source"]
            assert actual["line"] == recorded["line"]
            for field in ("policy_logits", "policy_probabilities", "value_prediction"):
                tolerance = 1e-6 if field == "policy_probabilities" else 1e-7
                assert np.allclose(
                    actual[field], recorded[field], rtol=0, atol=tolerance
                ), (
                    key,
                    field,
                )
    print("seed504 endpoint predictions and metrics independently reproduced")


if __name__ == "__main__":
    main()
