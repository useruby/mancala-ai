"""Synthetic endpoint publication tests using real model predictions."""

from __future__ import annotations

import json
import hashlib
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite import seed457_fresh_subgroup_projection as seed457
from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]


def fixture_data() -> tuple[
    dict[str, torch.nn.Module], tuple[np.ndarray, ...], list[dict[str, object]]
]:
    torch.manual_seed(4571)
    count = 512
    x = np.zeros((count, 27), dtype=np.float32)
    x[:, :6] = x[:, 6:12] = 1 / 48
    x[:, 12] = 1 / 48
    p = np.zeros((count, train.POLICY_SIZE), dtype=np.float32)
    p[:, 0] = 1
    p[1, 1] = 1
    v = np.linspace(-0.4, 0.4, count, dtype=np.float32)[:, None]
    replay = np.arange(count, dtype=np.int64)
    coefficients = np.ones(count, dtype=np.float32)
    positions = np.arange(count, dtype=np.int64)
    arrays = (
        x,
        p,
        v,
        replay,
        coefficients,
        positions,
        positions,
        [{} for _ in range(count)],
    )
    membership = [
        {
            "subset": "unseen",
            "active_stones": 40,
            "source": "fresh",
            "compact_row": i,
            "input_identity": f"id-{i // 2}",
        }
        for i in range(4)
    ]
    models = {
        label: train.PolicyValueNet((96, 3), "residual_v3", 27)
        for label in ("initializer", "C", "T")
    }
    return models, arrays, membership


def test_prediction_metrics_effects_and_equal_input_retain_all_exposures() -> None:
    models, arrays, membership = fixture_data()
    steps, tensors = {}, {}
    batches = np.arange(512, dtype=np.int64)
    guard_rows = np.arange(4, dtype=np.int64)
    shards = [[{"compact_row": i}] for i in range(8)]
    for arm in ("C", "T"):
        steps[arm], tensors[arm] = seed457._run_arm_engine(
            models[arm], arrays, batches, guard_rows, shards, arm, proposal_count=1
        )
    a_metrics = {
        w: {"policy_ce": 10.0, "value_mse": 0.0}
        for w in ("exposure_weighted", "equal_input")
    }
    out = ROOT / ".tmp" / f"seed457-publication-test-{uuid.uuid4().hex}"
    try:
        evidence = seed457.build_synthetic_publication(
            ROOT,
            models,
            arrays,
            membership,
            steps,
            tensors,
            seed442_a_metrics=a_metrics,
            initializer_training_objective=1.0,
            final_training_objectives={"initializer": 1.0, "C": 0.9, "T": 0.8},
            control={"matches": True, "mismatches": []},
            identities={
                "sources": {"synthetic-source": hashlib.sha256(b"source").hexdigest()},
                "inputs": {"synthetic-input": hashlib.sha256(b"input").hexdigest()},
            },
            output_dir=out,
        )
        assert evidence["status"] == "synthetic-test-fixture"
        archived = np.load(out / "endpoint-predictions.npz", allow_pickle=False)
        try:
            assert len(archived["T_fresh_identities"]) == 4
            assert archived["T_fresh_identities"].tolist() == [
                "id-0",
                "id-0",
                "id-1",
                "id-1",
            ]
            assert not np.array_equal(
                archived["T_fresh_targets"][0], archived["T_fresh_targets"][1]
            )
            equal = evidence["metrics"]["T"]["fresh"]["equal_input"]["policy_ce"]
            losses = []
            for i in range(4):
                z = torch.from_numpy(archived["T_fresh_logits"][i : i + 1])
                legal = torch.from_numpy(
                    train.legal_mask_matrix_for_encoded_states(
                        archived["T_fresh_inputs"][i : i + 1]
                    )
                )
                target = torch.from_numpy(archived["T_fresh_targets"][i : i + 1])
                losses.append(
                    float(
                        train.compute_policy_cross_entropy(
                            z.masked_fill(legal <= 0, -1e9), target
                        )[0]
                    )
                )
            assert equal == pytest.approx(
                (np.mean(losses[:2]) + np.mean(losses[2:])) / 2
            )
            assert evidence["effects_T_minus_C"]["equal_input"][
                "policy_ce"
            ] == pytest.approx(
                evidence["metrics"]["T"]["equal_input"]["policy_ce"]
                - evidence["metrics"]["C"]["equal_input"]["policy_ce"]
            )
        finally:
            archived.close()
        seed457.verify_synthetic_receipt(out)
        (out / "evidence.json").write_text(json.dumps(evidence))
        with pytest.raises(ValueError, match="hash_mismatch"):
            seed457.verify_synthetic_receipt(out)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_control_scale_or_tensor_mutation_is_reported() -> None:
    current = [{"selected_scale": 0.5}]
    archive = {
        f"C_00_{i:02d}_post": np.array([float(i)], dtype=np.float32) for i in range(22)
    }
    historical = {"arms": {"B": {"steps": [{"selected_scale": 0.5}]}}}
    old = {
        f"B_00_{i:02d}_post": np.array([float(i)], dtype=np.float32) for i in range(22)
    }
    assert seed457.compare_control(current, archive, historical, old)["matches"]
    old["B_00_03_post"][0] += 1
    result = seed457.compare_control(current, archive, historical, old)
    assert not result["matches"] and result["mismatches"] == [
        {"step": 1, "tensor": 3, "field": "post_parameters"}
    ]
    current[0]["selected_scale"] = 0.25
    assert any(
        item["field"] == "selected_scale"
        for item in seed457.compare_control(current, archive, historical, old)[
            "mismatches"
        ]
    )


@pytest.mark.parametrize(
    ("active", "gates", "expected"),
    [
        (False, False, "subgroup_projection_inactive_no_followup"),
        (True, False, "close_fresh_subgroup_projection_branch"),
        (True, True, "advance_to_separately_preregistered_strength_experiment"),
    ],
)
def test_decision_precedence(active: bool, gates: bool, expected: str) -> None:
    metric = {
        w: {"policy_ce": 0.0 if gates else 10.0, "value_mse": 0.0}
        for w in ("exposure_weighted", "equal_input")
    }
    reference = {
        w: {"policy_ce": 1.0 if gates else 10.0, "value_mse": 0.0}
        for w in ("exposure_weighted", "equal_input")
    }
    metrics = {
        "T": {**metric, "fresh": metric, "full_training_objective": 0.5},
        "initializer": {
            **reference,
            "fresh": reference,
            "full_training_objective": 1.0,
        },
        "seed442_A": reference,
    }
    steps = [
        {
            "proposal_norm": 1.0,
            "rejected": False,
            "additional_subgroup_protection": active,
        }
    ]
    assert seed457.decide(metrics, steps) == expected


def test_fixed_gate_boundaries_are_inclusive() -> None:
    pooled_t = {
        w: {"policy_ce": 0.995, "value_mse": 0.502}
        for w in ("exposure_weighted", "equal_input")
    }
    pooled_init = {w: {"policy_ce": 1.0, "value_mse": 0.5} for w in pooled_t}
    ordinary = {w: {"policy_ce": 1.0050001, "value_mse": 0.5} for w in pooled_t}
    fresh_t = {w: {"policy_ce": 0.200002, "value_mse": 0.0} for w in pooled_t}
    fresh_init = {w: {"policy_ce": 0.2, "value_mse": 0.0} for w in pooled_t}
    metrics = {
        "T": {**pooled_t, "fresh": fresh_t, "full_training_objective": 0.9},
        "initializer": {
            **pooled_init,
            "fresh": fresh_init,
            "full_training_objective": 1.0,
        },
        "seed442_A": ordinary,
    }
    gates = seed457.decision_gates(
        metrics,
        1.0,
        {"activated": True},
        [{"proposal_norm": 0.1, "rejected": False}],
    )
    assert all(gates.values())
