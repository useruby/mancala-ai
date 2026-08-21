from __future__ import annotations

import torch

from ml.alphazero_lite.run_fresh_p1_adapter_teacher_target_retrain import (
    cache_entry_is_valid,
    canonical_hash,
    lane_invariants,
    target_field,
)


def test_target_field_limits_matched_lanes_to_their_target_source() -> None:
    assert target_field("stored384") is None
    assert target_field("clean384") == "clean384"
    assert target_field("clean1200") == "clean1200"


def test_cache_entry_requires_matching_state_and_both_clean_budgets() -> None:
    row = {"state": [0.0] * 27}
    entry = {
        "schema": "azlite_pr214_full_replay_clean_target_v1",
        "state_hash": "bad",
        "clean384": {"policy": [1.0]},
        "clean1200": {"policy": [1.0]},
    }
    assert not cache_entry_is_valid(entry, row)

    entry["state_hash"] = canonical_hash(row["state"])
    assert cache_entry_is_valid(entry, row)


def test_lane_invariants_reject_non_adapter_drift() -> None:
    parent = {
        "policy_adapter.weight": torch.tensor([0.0]),
        "trunk.weight": torch.tensor([1.0]),
    }
    snapshots = {
        lane: {0: ({name: value.clone() for name, value in parent.items()}, {})}
        for lane in ("stored384", "clean384", "clean1200")
    }
    snapshots["clean1200"][0][0]["trunk.weight"] += 1

    assert not lane_invariants(snapshots, parent)[
        "non_adapter_parameters_bit_identical"
    ]
