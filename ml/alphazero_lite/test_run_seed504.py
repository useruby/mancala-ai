from __future__ import annotations

import numpy as np
import torch

from ml.alphazero_lite import run_seed504, train


def test_split_excludes_both_canonical_and_float32_identity() -> None:
    training, development, evidence = run_seed504.split()
    train_canonical = {run_seed504.keys(row)[0] for row in training}
    train_inputs = {run_seed504.keys(row)[1] for row in training}
    assert all(run_seed504.keys(row)[0] not in train_canonical for row in development)
    assert all(run_seed504.keys(row)[1] not in train_inputs for row in development)
    assert evidence["counts"]["remaining_overlap_canonical"] == 0
    assert evidence["counts"]["remaining_overlap_float32"] == 0


def test_production_loader_preserves_training_rows_and_unit_source_weights() -> None:
    paths = [run_seed504.CORPUS / f"shard-{i}.jsonl" for i in range(503000, 503006)]
    x, _, _, replay, policy_weights = train.load_jsonl_replay(
        paths, weights=[1] * 6, include_policy_loss_weights=True
    )
    training, _, _ = run_seed504.split()
    expected = np.asarray([row["state"] for row in training], dtype=np.float32)
    np.testing.assert_array_equal(x, expected)
    np.testing.assert_array_equal(replay, np.arange(len(training)))
    np.testing.assert_array_equal(
        policy_weights, np.ones(len(training), dtype=np.float32)
    )


def test_equal_exact_input_preserves_distinct_repeated_targets() -> None:
    state = (
        [1 / 12] * 12 + [0.0] * 3 + [1.0] + [0.0] * 2 + [1 / 6, 0.5, 1.0] + [0.0] * 6
    )
    rows = [
        {
            "state": state,
            "policy": [1, 0, 0, 0, 0, 0],
            "value": 0.0,
            "_source": "a",
            "_line": 1,
        },
        {
            "state": state,
            "policy": [0, 1, 0, 0, 0, 0],
            "value": 1.0,
            "_source": "a",
            "_line": 2,
        },
    ]
    model = train.PolicyValueNet((8, 3), "residual_v3", 27).eval()
    result, _ = run_seed504.evaluate(model, rows)
    assert result["pooled"]["rows"] == 2
    assert result["pooled"]["exact_inputs"] == 1
    _, predictions = run_seed504.evaluate(model, rows)
    assert [row["target_value"] for row in predictions] == [0.0, 1.0]


def test_exposure_and_equal_input_metric_formulas() -> None:
    _, development, _ = run_seed504.split()
    grouped: dict[tuple[float, ...], list[dict]] = {}
    for row in development:
        grouped.setdefault(run_seed504.keys(row)[1], []).append(row)
    repeated = next(group for group in grouped.values() if len(group) >= 2)
    duplicate = repeated[:2]
    alternate = [0.0] * 6
    alternate[duplicate[1]["legal_moves"][0]] = 1.0
    duplicate[1] = {**duplicate[1], "policy": alternate, "value": -0.75}
    singleton = next(group[0] for group in grouped.values() if len(group) == 1)
    rows = [*duplicate, singleton]
    model = train.PolicyValueNet((8, 3), "residual_v3", 27).eval()
    metrics, predictions = run_seed504.evaluate(model, rows)
    per_row_ce = []
    per_row_mse = []
    for row, prediction in zip(rows, predictions, strict=True):
        policy = np.asarray(row["policy"], dtype=np.float64)
        probs = np.asarray(prediction["policy_probabilities"], dtype=np.float64)
        per_row_ce.append(float(-(policy * np.log(np.maximum(probs, 1e-300))).sum()))
        per_row_mse.append((prediction["value_prediction"] - row["value"]) ** 2)
    expected_equal_ce = (np.mean(per_row_ce[:2]) + per_row_ce[2]) / 2
    expected_equal_mse = (np.mean(per_row_mse[:2]) + per_row_mse[2]) / 2
    assert np.isclose(metrics["pooled"]["exposure"]["policy_ce"], np.mean(per_row_ce))
    assert np.isclose(
        metrics["pooled"]["equal_exact_input"]["policy_ce"], expected_equal_ce
    )
    assert np.isclose(
        metrics["pooled"]["exposure"]["actual_value_mse"], np.mean(per_row_mse)
    )
    assert np.isclose(
        metrics["pooled"]["equal_exact_input"]["actual_value_mse"], expected_equal_mse
    )


def test_checkpoint_loader_and_fixed_permutation_shape() -> None:
    model = run_seed504.load_model(run_seed504.INITIALIZER, torch.device("cpu"))
    logits, value = model(torch.zeros((1, 27)))
    assert logits.shape == (1, 6)
    assert value.shape == (1, 1)
    generator = torch.Generator(device="cpu").manual_seed(504)
    order = torch.randperm(13724, generator=generator).tolist()
    assert len(order) == 13724
    assert len(set(order)) == 13724
