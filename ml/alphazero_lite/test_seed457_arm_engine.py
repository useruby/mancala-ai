"""Real-model synthetic checks for the seed457 arm update engine."""

from __future__ import annotations

import numpy as np
import torch

from ml.alphazero_lite import seed457_fresh_subgroup_projection as seed457
from ml.alphazero_lite import train


def context() -> tuple[
    torch.nn.Module,
    tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
    list[list[dict[str, object]]],
]:
    torch.manual_seed(457)
    count = 512
    states = np.zeros((count, 27), dtype=np.float32)
    states[:, :6] = 1.0 / 48.0
    states[:, 6:12] = 1.0 / 48.0
    states[:, 12] = 1.0 / 48.0
    policy = np.zeros((count, train.POLICY_SIZE), dtype=np.float32)
    policy[:, 0] = 1.0
    values = np.linspace(-0.5, 0.5, count, dtype=np.float32)[:, None]
    replay = np.arange(count, dtype=np.int64)
    coefficients = np.ones(count, dtype=np.float32)
    train_positions = np.arange(count, dtype=np.int64)
    arrays = (
        states,
        policy,
        values,
        replay,
        coefficients,
        train_positions,
        train_positions,
        [{} for _ in range(count)],
    )
    batches = np.arange(512, dtype=np.int64)
    full_guard = np.arange(4, dtype=np.int64)
    shards = [[{"compact_row": index}] for index in range(8)]
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    return model, arrays, batches, full_guard, shards


def snapshot(model: torch.nn.Module) -> list[np.ndarray]:
    return [parameter.detach().cpu().numpy().copy() for parameter in model.parameters()]


def test_actual_engine_accepts_largest_feasible_update_and_advances_adam_once() -> None:
    model, arrays, batches, guard, shards = context()
    before = snapshot(model)
    records, archive = seed457._run_arm_engine(
        model, arrays, batches, guard, shards, "C", proposal_count=1
    )
    record = records[0]
    assert len(record["trials"]) == 8
    assert record["optimizer_step"] == 1
    assert record["objective_gradient_preserved"]
    assert record["selected_scale"] == next(
        t["scale"] for t in record["trials"] if t["feasible"]
    )
    assert sum(bool(t["feasible"]) for t in record["trials"]) >= 1
    for scale in seed457.seed453.SCALES:
        expected = seed457.trial_state(
            [archive[f"C_00_{i:02d}_pre"] for i in range(len(before))],
            [archive[f"C_00_{i:02d}_projected_proposal"] for i in range(len(before))],
            scale,
        )
        actual = [archive[f"C_00_{i:02d}_trial_{scale:g}"] for i in range(len(before))]
        assert all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))
    selected_arrays = [
        archive[f"C_00_{i:02d}_trial_{record['selected_scale']:g}"]
        for i in range(len(before))
    ]
    assert all(
        np.array_equal(a, b)
        for a, b in zip(snapshot(model), selected_arrays, strict=True)
    )
    assert any(
        not np.array_equal(a, b) for a, b in zip(before, snapshot(model), strict=True)
    )
    for i in range(len(before)):
        prefix = f"C_00_{i:02d}"
        gradient = archive[f"{prefix}_objective_gradient"]
        np.testing.assert_array_equal(archive[f"{prefix}_exp_avg_pre"], 0)
        np.testing.assert_array_equal(archive[f"{prefix}_exp_avg_sq_pre"], 0)
        np.testing.assert_allclose(
            archive[f"{prefix}_exp_avg"], 0.1 * gradient, rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            archive[f"{prefix}_exp_avg_sq"], 0.001 * gradient**2, rtol=0, atol=1e-8
        )


def test_actual_engine_rejects_nonzero_proposal_but_retains_adam_update() -> None:
    model, arrays, batches, guard, shards = context()
    before = snapshot(model)
    records, archive = seed457._run_arm_engine(
        model,
        arrays,
        batches,
        guard,
        shards,
        "C",
        proposal_count=1,
        limits=seed457._ConstraintLimits(kl=-1.0),
    )
    row = records[0]
    assert row["rejected"] and row["selected_scale"] is None
    assert not any(trial["feasible"] for trial in row["trials"])
    assert row["proposal_norm"] > 0
    assert all(
        np.array_equal(a, b) for a, b in zip(before, snapshot(model), strict=True)
    )
    assert any(
        np.any(archive[f"C_00_{i:02d}_exp_avg"] != 0) for i in range(len(before))
    )
    assert any(
        not np.array_equal(
            archive[f"C_00_{i:02d}_exp_avg"], archive[f"C_00_{i:02d}_exp_avg_pre"]
        )
        for i in range(len(before))
    )


def test_actual_engine_archives_are_defensive_copies() -> None:
    model, arrays, batches, guard, shards = context()
    _records, archive = seed457._run_arm_engine(
        model, arrays, batches, guard, shards, "T", proposal_count=1
    )
    keys = (
        "pre",
        "post",
        "raw_proposal",
        "projected_proposal",
        "objective_gradient",
        "fresh_gradient",
        "exp_avg",
        "exp_avg_sq",
    )
    frozen = {key: archive[f"T_00_00_{key}"].copy() for key in keys}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    assert all(
        np.array_equal(archive[f"T_00_00_{key}"], value)
        for key, value in frozen.items()
    )
