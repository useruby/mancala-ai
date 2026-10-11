"""Full linked-trajectory tests with real seed457 engine/model calculations."""

from __future__ import annotations

import copy
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
    reconstruct_trajectory,
)

ROOT = Path(__file__).resolve().parents[2]


def make_context() -> tuple[
    torch.nn.Module,
    tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
    list[list[dict[str, int]]],
]:
    torch.manual_seed(45716)
    count = 16 * 512
    states = np.zeros((count, 27), dtype=np.float32)
    states[:, :6] = states[:, 6:12] = 1 / 48
    states[:, 12] = 1 / 48
    policy = np.zeros((count, train.POLICY_SIZE), dtype=np.float32)
    policy[:, 0] = 1
    values = np.linspace(-0.4, 0.4, count, dtype=np.float32)[:, None]
    replay = np.arange(count, dtype=np.int64)
    coeff = np.ones(count, dtype=np.float32)
    positions = np.arange(count, dtype=np.int64)
    arrays = (
        states,
        policy,
        values,
        replay,
        coeff,
        positions,
        positions,
        [{} for _ in range(count)],
    )
    batches = np.arange(count, dtype=np.int64)
    full_rows = np.arange(4, dtype=np.int64)
    shards = [[{"compact_row": i}] for i in range(8)]
    return (
        train.PolicyValueNet((96, 3), "residual_v3", 27),
        arrays,
        batches,
        full_rows,
        shards,
    )


def execute(arm: str, *, rejected: bool = False):
    model, arrays, batches, full_rows, shards = make_context()
    initializer = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    limits = runner._ConstraintLimits(kl=-1.0) if rejected else None
    records, archive = runner._run_arm_engine(
        model,
        arrays,
        batches,
        full_rows,
        shards,
        arm,
        limits=limits,
        fresh_guard_rows=np.arange(8),
    )
    artifact_dir = ROOT / ".tmp" / f"seed457-trajectory-{uuid.uuid4().hex}"
    artifact_dir.mkdir(parents=True)
    np.savez_compressed(artifact_dir / f"{arm}-steps.npz", **archive)
    return (
        model,
        arrays,
        batches,
        full_rows,
        shards,
        initializer,
        records,
        archive,
        artifact_dir,
        rejected,
        arm,
    )


def verify_fixture(fixture) -> dict:
    (
        _model,
        arrays,
        batches,
        full_rows,
        shards,
        initializer,
        records,
        archive,
        _path,
        rejected,
        arm,
    ) = fixture
    return reconstruct_trajectory(
        model=train.PolicyValueNet((96, 3), "residual_v3", 27),
        initializer=initializer,
        arrays=arrays,
        arm=arm,
        records=records,
        archive=archive,
        batches=batches,
        full_guard_rows=full_rows,
        fresh_guard_rows=np.arange(8),
        subgroup_rows=[np.asarray([r["compact_row"] for r in sh]) for sh in shards],
        fixture_limits=(-1.0, 0.0001, 1e-7) if rejected else None,
    )


@pytest.mark.parametrize(
    ("arm", "rejected"), [("C", False), ("T", False), ("C", True), ("T", True)]
)
def test_valid_sixteen_step_accepted_and_rejected_trajectories(
    arm: str, rejected: bool
) -> None:
    fixture = execute(arm, rejected=rejected)
    try:
        # Arm identity is supplied independently; the current engine record is made explicit.
        facts = verify_fixture(fixture)
        assert facts["steps"] == 16 and facts["optimizer_step"] == 16
        assert facts["rejected_count"] == (16 if rejected else 0)
        assert len(facts["final_parameters"]) == 22
        if rejected:
            for i, initial in enumerate(fixture[5]):
                np.testing.assert_array_equal(facts["final_parameters"][i], initial)
            assert any(np.any(a != 0) for a, _ in facts["final_moments"])
    finally:
        shutil.rmtree(fixture[8], ignore_errors=True)


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        ("missing", "trajectory_record_count_invalid"),
        ("duplicate", "trajectory_step_order_invalid"),
        ("reordered", "trajectory_step_order_invalid"),
        ("number", "trajectory_step_order_invalid"),
        ("minibatch", "minibatch_identity_order_invalid"),
        ("initializer", "pre_trajectory"),
        ("pre", "pre_trajectory"),
        ("post", "post_state"),
        ("moment", "pre_moment|moment:"),
        ("endpoint", "post_state"),
    ],
)
def test_trajectory_reconstruction_rejects_link_and_order_mutations(
    mutation: str, failure: str
) -> None:
    fixture = execute("C")
    try:
        records = copy.deepcopy(fixture[6])
        archive = {key: value.copy() for key, value in fixture[7].items()}
        rejected = fixture[9]
        if mutation == "missing":
            records.pop()
        elif mutation == "duplicate":
            records[5] = copy.deepcopy(records[4])
        elif mutation == "reordered":
            records[0], records[1] = records[1], records[0]
        elif mutation == "number":
            records[4]["step"] = 9
        elif mutation == "minibatch":
            records[3]["compact_rows"][0] += 1
        elif mutation == "initializer":
            fixture[5][0].flat[0] += 0.1
        elif mutation == "pre":
            archive["C_00_00_pre"].flat[0] += 0.1
        elif mutation == "post":
            archive["C_05_00_post"].flat[0] += 0.1
        elif mutation == "moment":
            archive["C_05_00_exp_avg"].flat[0] += 0.1
        else:
            archive["C_15_00_post"].flat[0] += 0.1
        with pytest.raises(ValueError, match=failure):
            reconstruct_trajectory(
                model=train.PolicyValueNet((96, 3), "residual_v3", 27),
                initializer=fixture[5],
                arrays=fixture[1],
                arm=fixture[10],
                records=records,
                archive=archive,
                batches=fixture[2],
                full_guard_rows=fixture[3],
                fresh_guard_rows=np.arange(8),
                subgroup_rows=[
                    np.asarray([r["compact_row"] for r in sh]) for sh in fixture[4]
                ],
                fixture_limits=(-1, 0.0001, 1e-7) if rejected else None,
            )
    finally:
        shutil.rmtree(fixture[8], ignore_errors=True)


def test_c_reference_compares_all_sixteen_scales_and_post_tensors() -> None:
    fixture = execute("C")
    try:
        records = fixture[6]
        archive = fixture[7]
        reference_records = [
            {"selected_scale": row["selected_scale"]} for row in records
        ]
        reference_archive = {
            f"B_{s:02d}_{i:02d}_post": archive[f"C_{s:02d}_{i:02d}_post"].copy()
            for s in range(16)
            for i in range(22)
        }
        # The reference comparison contract accepts full protocol records.
        reference_records = [dict(row) for row in records]
        assert compare_c_reference(
            records, archive, reference_records, reference_archive
        )["matches"]
        reference_archive["B_09_03_post"].flat[0] += 1
        mismatch = compare_c_reference(
            records, archive, reference_records, reference_archive
        )
        assert mismatch["mismatches"] == [
            {"step": 10, "tensor": 3, "field": "post_parameters"}
        ]
        reference_records[2]["selected_scale"] = 0.125
        assert any(
            x["field"] == "selected_scale"
            for x in compare_c_reference(
                records, archive, reference_records, reference_archive
            )["mismatches"]
        )
    finally:
        shutil.rmtree(fixture[8], ignore_errors=True)
