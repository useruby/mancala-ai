"""End-to-end synthetic publication verification using real engine/model work."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite import seed457_fresh_subgroup_projection as runner
from ml.alphazero_lite import train
from ml.alphazero_lite.verify_seed457_fresh_subgroup_projection import (
    compare_c_reference,
    verify_synthetic_publication,
)

ROOT = Path(__file__).resolve().parents[2]


def make_full_fixture():
    torch.manual_seed(45716)
    count = 8192
    x = np.zeros((count, 27), dtype=np.float32)
    x[:, :6] = x[:, 6:12] = 1 / 48
    x[:, 12] = 1 / 48
    policy = np.zeros((count, train.POLICY_SIZE), dtype=np.float32)
    policy[:, 0] = 1
    values = np.linspace(-0.4, 0.4, count, dtype=np.float32)[:, None]
    replay = np.arange(count, dtype=np.int64)
    coefficient = np.ones(count, dtype=np.float32)
    positions = np.arange(count, dtype=np.int64)
    arrays = (
        x,
        policy,
        values,
        replay,
        coefficient,
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
    base = train.PolicyValueNet((96, 3), "residual_v3", 27)
    initial = [p.detach().numpy().copy() for p in base.parameters()]
    models = {"initializer": base}
    for arm in ("C", "T"):
        current = train.PolicyValueNet((96, 3), "residual_v3", 27)
        with torch.no_grad():
            for p, value in zip(current.parameters(), initial, strict=True):
                p.copy_(torch.from_numpy(value.copy()))
        models[arm] = current
    batches = np.arange(count, dtype=np.int64)
    full_rows = np.arange(4, dtype=np.int64)
    shards = [[{"compact_row": i}] for i in range(8)]
    return models, arrays, membership, initial, batches, full_rows, shards


def expanded_objective(model, arrays) -> float:
    x, policy, target, replay, coefficient, positions, *_ = arrays
    rows = replay[positions]
    policy_losses, huber_losses = [], []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(x), 512):
            logits, predicted = model(torch.from_numpy(x[begin : begin + 512]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[begin : begin + 512])
            )
            policy_losses.extend(
                train.compute_policy_cross_entropy(
                    logits.masked_fill(legal <= 0, -1e9),
                    torch.from_numpy(policy[begin : begin + 512]),
                )
                .double()
                .numpy()
            )
            huber_losses.extend(
                train.compute_value_loss_vector(
                    predicted,
                    torch.from_numpy(target[begin : begin + 512]),
                    value_loss="huber",
                    huber_delta=1,
                )
                .double()
                .numpy()
            )
    p = np.asarray(policy_losses)[rows]
    h = np.asarray(huber_losses)[rows]
    return float(np.average(p, weights=coefficient[rows]) + 0.3 * np.mean(h))


def publication_fixture():
    models, arrays, members, initial, batches, full_rows, shards = make_full_fixture()
    records, archives = {}, {}
    for arm in ("C", "T"):
        records[arm], archives[arm] = runner._run_arm_engine(
            models[arm],
            arrays,
            batches,
            full_rows,
            shards,
            arm,
            fresh_guard_rows=np.arange(8),
        )
    reference_records = records["C"]
    reference_arrays = {
        f"B_{s:02d}_{i:02d}_post": archives["C"][f"C_{s:02d}_{i:02d}_post"].copy()
        for s in range(16)
        for i in range(22)
    }
    control = compare_c_reference(
        records["C"], archives["C"], reference_records, reference_arrays
    )
    root_out = ROOT / ".tmp" / f"seed457-full-verification-{uuid.uuid4().hex}"
    ordinary = {
        w: {"policy_ce": 10.0, "value_mse": 0.0}
        for w in ("exposure_weighted", "equal_input")
    }
    objectives = {
        name: expanded_objective(model, arrays) for name, model in models.items()
    }
    runner.build_synthetic_publication(
        ROOT,
        models,
        arrays,
        members,
        records,
        archives,
        seed442_a_metrics=ordinary,
        initializer_training_objective=objectives["initializer"],
        final_training_objectives=objectives,
        control=control,
        identities={
            "sources": {"synthetic": hashlib.sha256(b"synthetic source").hexdigest()},
            "inputs": {"synthetic": hashlib.sha256(b"synthetic input").hexdigest()},
        },
        output_dir=root_out,
    )
    return (
        root_out,
        arrays,
        members,
        initial,
        batches,
        full_rows,
        shards,
        records,
        archives,
        objectives,
    )


def test_complete_synthetic_publication_verifies() -> None:
    (
        out,
        arrays,
        members,
        initial,
        batches,
        full_rows,
        shards,
        records,
        archives,
        _objectives,
    ) = publication_fixture()
    try:
        reference = {
            f"B_{s:02d}_{i:02d}_post": archives["C"][f"C_{s:02d}_{i:02d}_post"]
            for s in range(16)
            for i in range(22)
        }
        result = verify_synthetic_publication(
            root=ROOT,
            output_dir=out,
            arrays=arrays,
            membership=members,
            initializer=initial,
            model=train.PolicyValueNet((96, 3), "residual_v3", 27),
            batches=batches,
            full_guard_rows=full_rows,
            fresh_guard_rows=np.arange(8),
            subgroup_rows=[np.asarray([i]) for i in range(8)],
            reference_records=records["C"],
            reference_archive=reference,
        )
        assert result["status"] == "valid_synthetic_fixture"
        prediction_path = out / "endpoint-predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as prediction_data:
            changed = {
                key: prediction_data[key].copy() for key in prediction_data.files
            }
        changed["T_fresh_logits"].flat[0] += 0.25
        np.savez_compressed(prediction_path, **changed)
        receipt_path = out / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["files_sha256"][prediction_path.name] = hashlib.sha256(
            prediction_path.read_bytes()
        ).hexdigest()
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        with pytest.raises(ValueError, match="endpoint_prediction:T_fresh_logits"):
            verify_synthetic_publication(
                root=ROOT,
                output_dir=out,
                arrays=arrays,
                membership=members,
                initializer=initial,
                model=train.PolicyValueNet((96, 3), "residual_v3", 27),
                batches=batches,
                full_guard_rows=full_rows,
                fresh_guard_rows=np.arange(8),
                subgroup_rows=[np.asarray([i]) for i in range(8)],
                reference_records=records["C"],
                reference_archive=reference,
            )
    finally:
        shutil.rmtree(out, ignore_errors=True)
