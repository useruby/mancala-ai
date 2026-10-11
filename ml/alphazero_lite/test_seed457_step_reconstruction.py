"""Real-computation semantic tests for independent seed457 step replay."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from ml.alphazero_lite import seed457_fresh_subgroup_projection as runner
from ml.alphazero_lite import verify_seed457_fresh_subgroup_projection as verifier
from ml.alphazero_lite.test_seed457_arm_engine import context


@pytest.mark.parametrize("arm", ["C", "T"])
@pytest.mark.parametrize("reject", [False, True])
def test_independently_reconstructs_actual_engine_step(arm: str, reject: bool) -> None:
    model, arrays, batches, full_guard, shards = context()
    limits = runner._ConstraintLimits(kl=-1.0) if reject else None
    records, archive = runner._run_arm_engine(
        model, arrays, batches, full_guard, shards, arm, proposal_count=1, limits=limits
    )
    result = verifier.reconstruct_engine_step(
        model=model,
        arrays=arrays,
        arm=arm,
        record=records[0],
        archive=archive,
        full_guard_rows=full_guard,
        fresh_guard_rows=np.asarray(
            [row["compact_row"] for shard in shards for row in shard]
        ),
        subgroup_rows=[
            np.asarray([row["compact_row"] for row in shard]) for shard in shards
        ],
        fixture_limits=(-1.0, 0.0001, 1e-7) if reject else None,
    )
    assert result["verified_trials"] == 8
    assert result["rejected"] is reject


@pytest.mark.parametrize(
    ("target", "category"),
    [
        ("objective_gradient", "clipped_gradient"),
        ("exp_avg", "moment:"),
        ("raw_proposal", "raw_proposal:"),
        ("trial", "trial_parameters:"),
        ("post", "post_state:"),
    ],
)
def test_reconstruction_rejects_altered_step_tensor(target: str, category: str) -> None:
    model, arrays, batches, full_guard, shards = context()
    records, original = runner._run_arm_engine(
        model, arrays, batches, full_guard, shards, "C", proposal_count=1
    )
    archive = {key: value.copy() for key, value in original.items()}
    key = (
        f"C_00_00_trial_{runner.seed453.SCALES[0]:g}"
        if target == "trial"
        else f"C_00_00_{target}"
    )
    archive[key].flat[0] += 0.125
    model2, arrays2, _, guard2, shards2 = context()
    with pytest.raises(ValueError, match=category):
        verifier.reconstruct_engine_step(
            model=model2,
            arrays=arrays2,
            arm="C",
            record=copy.deepcopy(records[0]),
            archive=archive,
            full_guard_rows=guard2,
            fresh_guard_rows=np.asarray(
                [r["compact_row"] for sh in shards2 for r in sh]
            ),
            subgroup_rows=[
                np.asarray([r["compact_row"] for r in sh]) for sh in shards2
            ],
        )


def test_reconstruction_rejects_measurement_and_scale_tampering() -> None:
    model, arrays, batches, guard, shards = context()
    records, archive = runner._run_arm_engine(
        model, arrays, batches, guard, shards, "C", proposal_count=1
    )
    model2, arrays2, _, guard2, shards2 = context()
    changed = copy.deepcopy(records[0])
    changed["trials"][0]["batch_kl"] += 0.1
    kwargs = dict(
        model=model2,
        arrays=arrays2,
        arm="C",
        record=changed,
        archive=archive,
        full_guard_rows=guard2,
        fresh_guard_rows=np.arange(8),
        subgroup_rows=[np.asarray([i]) for i in range(8)],
    )
    with pytest.raises(ValueError, match="trial_measurement_mismatch"):
        verifier.reconstruct_engine_step(**kwargs)
    changed = copy.deepcopy(records[0])
    changed["selected_scale"] = 1 / 128 if changed["selected_scale"] != 1 / 128 else 1.0
    kwargs["record"] = changed
    with pytest.raises(ValueError, match="selected_scale_mismatch"):
        verifier.reconstruct_engine_step(**kwargs)
