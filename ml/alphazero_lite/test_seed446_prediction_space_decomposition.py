"""Focused arithmetic tests for seed446's frozen prediction-space identity."""

import numpy as np
import pytest

from ml.alphazero_lite.seed446_prediction_space_decomposition import (
    classify,
    decompose,
    weights,
)
from ml.alphazero_lite.verify_seed446_prediction_space_decomposition import (
    ARMS,
    WEIGHTINGS,
    validate_protocol,
    validate_semantics,
)


def test_linear_prediction_has_zero_remainder() -> None:
    before = np.array([0.2, -0.4], dtype=np.float32)
    after = np.array([0.3, -0.2], dtype=np.float32)
    target = np.array([0.0, 0.5], dtype=np.float32)
    displacement = np.array([0.1, 0.2], dtype=np.float64)
    gradient = np.array([-1.6, 0.0], dtype=np.float64)
    result = decompose(before, after, target, np.ones(2), gradient, displacement)
    assert result["n"] == pytest.approx(0.0, abs=2e-8)
    assert result["d"] == pytest.approx(result["c"] + result["q"])


@pytest.mark.parametrize("n", [0.2, -0.2])
def test_n_can_have_either_sign(n: float) -> None:
    before = np.zeros(2, dtype=np.float32)
    after = np.array([0.2, 0.1], dtype=np.float32)
    target = np.full(2, -1.0, dtype=np.float32)
    gradient = np.array([0.0, 0.0] if n > 0 else [4.0, 0.0], dtype=np.float64)
    displacement = np.array([0.1, 0.1], dtype=np.float64)
    result = decompose(before, after, target, np.ones(2), gradient, displacement)
    assert (result["n"] > 0) if n > 0 else (result["n"] < 0)


def test_zero_movement_and_duplicate_exposure_weighting() -> None:
    identity_weights = weights(["x", "x", "y"])
    assert identity_weights["exposure_weighted"] == pytest.approx([1 / 3] * 3)
    assert identity_weights["equal_input"] == pytest.approx([1 / 4, 1 / 4, 1 / 2])
    result = decompose(
        np.array([1, 2, 3], dtype=np.float32),
        np.array([1, 2, 3], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        identity_weights["equal_input"],
        np.zeros(1),
        np.zeros(1),
    )
    assert result["c"] == result["q"] == result["n"] == result["d"] == 0


def test_both_weightings_keep_duplicate_exposures() -> None:
    weight_vectors = weights(["same", "same", "other"])
    predictions = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    target = np.zeros(3, dtype=np.float32)
    displacement = np.array([0.2], dtype=np.float64)
    gradient = np.array([0.0], dtype=np.float64)
    values = [
        decompose(
            np.zeros(3, dtype=np.float32),
            predictions,
            target,
            vector,
            gradient,
            displacement,
        )["q"]
        for vector in weight_vectors.values()
    ]
    assert values[0] == pytest.approx(4.0 / 3.0)
    assert values[1] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("q", "n", "expected"),
    [
        (3.0, 1.0, "squared_prediction_movement_dominant"),
        (1.0, 3.0, "output_response_remainder_dominant"),
        (1.0, 1.0, "mixed_or_no_common_residual_component"),
        (1.0, -1.0, "mixed_or_no_common_residual_component"),
        (3.0, 1.0 + 1e-10, "mixed_or_no_common_residual_component"),
        (0.0, 0.0, "mixed_or_no_common_residual_component"),
    ],
)
def test_classifier_branches_and_boundaries(q: float, n: float, expected: str) -> None:
    assert classify(q, n) == expected


def semantic_fixture():
    targets = np.array([0.0, 1.0], dtype=np.float32)
    compact_rows = [3, 8]
    identities = ["a", "b"]
    predictions = {}
    gradients = {}
    states: dict[str, list[list[np.ndarray]]] = {}
    seed445 = {"arms": {}}
    published_rows = []
    totals = {arm: {} for arm in ARMS}
    for arm in ARMS:
        states[arm] = [
            [
                np.array([step * 0.1], dtype=np.float32)
                if index == 0
                else np.zeros(1, dtype=np.float32)
                for index in range(22)
            ]
            for step in range(17)
        ]
        seed445["arms"][arm] = {"step_ledger": []}
        for step in range(17):
            predictions[f"{arm}_{step:02d}_values"] = np.zeros(2, dtype=np.float32)
        for weighting in WEIGHTINGS:
            totals[arm][weighting] = {
                "c": 0.0,
                "q": 0.0,
                "s": 1.6,
                "n": -1.6,
                "d": 0.0,
            }
            for step in range(16):
                gradient = np.zeros(22)
                gradient[0] = 1.0
                gradients[f"{arm}_value_mse_{weighting}_{step:02d}"] = gradient
                seed445["arms"][arm]["step_ledger"].append(
                    {
                        "step": step + 1,
                        "weighting": weighting,
                        "objective": "value_mse",
                        "d": 0.0,
                        "s": 0.1,
                    }
                )
                published_rows.append(
                    {
                        "arm": arm,
                        "weighting": weighting,
                        "step": step + 1,
                        "c": 0.0,
                        "q": 0.0,
                        "s": 0.1,
                        "n": -0.1,
                        "d": 0.0,
                    }
                )
    layout = [
        {
            "name": f"p{index}",
            "shape": [1],
            "start": index,
            "stop": index + 1,
            "group": "value_head",
        }
        for index in range(22)
    ]
    published_targets = {
        "compact_row": list(compact_rows),
        "input_identity": list(identities),
        "value_target": targets.tolist(),
    }
    published = {
        "rows": published_rows,
        "totals": totals,
        "weighting_classifications": {
            weighting: "mixed_or_no_common_residual_component"
            for weighting in WEIGHTINGS
        },
        "classification": "mixed_or_no_common_residual_component",
    }
    return (
        targets,
        compact_rows,
        identities,
        predictions,
        gradients,
        states,
        layout,
        seed445,
        published_targets,
        published,
    )


def test_semantic_validator_accepts_valid_fixture_and_signed_cancellation() -> None:
    values = semantic_fixture()
    assert validate_semantics(*values)["status"] == "valid"
    # Signed cancellation can make a component share exceed one; no clipping is applied.
    assert classify(2.0, -1.0) == "squared_prediction_movement_dominant"
    assert 2.0 / (2.0 - 1.0) > 1.0


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("target", "published_targets_mismatch"),
        ("row", "published_compact_rows_mismatch"),
        ("identity", "published_identities_mismatch"),
        ("prediction", "seed445_dot_or_displacement_mismatch"),
        ("displacement", "seed445_dot_or_displacement_mismatch"),
        ("gradient", "seed445_dot_or_displacement_mismatch"),
        ("dot", "seed445_dot_or_displacement_mismatch"),
        ("missing_source_step", "seed445_ledger_coverage_invalid"),
        ("duplicate_source_step", "seed445_ledger_coverage_invalid"),
        ("missing", "published_ledger_coverage_invalid"),
        ("duplicate", "published_ledger_identity_invalid"),
        ("total", "published_total_q_invalid"),
        ("classification", "classification_invalid"),
    ],
)
def test_semantic_validator_rejects_tampering(tamper: str, message: str) -> None:
    import copy

    values: list = list(copy.deepcopy(semantic_fixture()))
    if tamper == "target":
        values[8]["value_target"][0] = 4.0
    elif tamper == "row":
        values[1][0] = 5
    elif tamper == "identity":
        values[2][0] = "wrong"
    elif tamper == "prediction":
        values[3]["A_01_values"][0] = 0.25
    elif tamper == "displacement":
        values[5]["A"][1][0][0] = 0.25
    elif tamper == "gradient":
        values[4]["A_value_mse_exposure_weighted_00"][0] = 2.0
    elif tamper == "dot":
        values[7]["arms"]["A"]["step_ledger"][0]["s"] = 0.5
    elif tamper == "missing_source_step":
        values[7]["arms"]["A"]["step_ledger"].pop()
    elif tamper == "duplicate_source_step":
        values[7]["arms"]["A"]["step_ledger"][1]["step"] = 1
    elif tamper == "missing":
        values[9]["rows"].pop()
    elif tamper == "duplicate":
        values[9]["rows"][1] = copy.deepcopy(values[9]["rows"][0])
    elif tamper == "total":
        values[9]["totals"]["B"]["equal_input"]["q"] = 1.0
    elif tamper == "classification":
        values[9]["classification"] = "squared_prediction_movement_dominant"
    with pytest.raises(ValueError, match=message):
        validate_semantics(*values)


def test_protocol_rejects_altered_weighting_definition() -> None:
    import json
    from pathlib import Path

    protocol = json.loads(
        Path(
            "docs/data/seed446-prediction-space-decomposition/protocol.json"
        ).read_text()
    )
    protocol["weightings"]["equal_input"] = "incorrect"
    with pytest.raises(ValueError, match="weighting_definition_invalid"):
        validate_protocol(protocol)
