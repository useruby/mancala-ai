"""Synthetic integration checks for seed457 preparation and verification."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from ml.alphazero_lite.seed457_fresh_subgroup_projection import shard_guard
from ml.alphazero_lite.verify_seed457_fresh_subgroup_projection import (
    partition,
    verify_evidence_structure,
    verify_kkt,
)


def synthetic_guard() -> list[dict[str, object]]:
    return [
        {"exact_input_identity": f"identity-{index:04d}", "compact_row": index}
        for index in range(1399)
    ]


def test_runner_and_verifier_partition_agree_and_preserve_order() -> None:
    guard = synthetic_guard()
    actual = shard_guard(guard)
    independent = partition(guard)
    assert actual == independent
    assert sorted(map(len, actual)) == [174] + [175] * 7
    for shard in actual:
        assert [row["compact_row"] for row in shard] == sorted(
            row["compact_row"] for row in shard
        )


def test_partition_rejects_duplicate_membership() -> None:
    guard = synthetic_guard()
    guard[-1]["exact_input_identity"] = guard[0]["exact_input_identity"]
    with pytest.raises(ValueError, match="identity_duplicate|population_invalid"):
        shard_guard(guard)


def test_independent_kkt_reconstruction_and_tamper() -> None:
    d = [np.array([1.0, 2.0])]
    gradients = [[np.array([1.0, 0.0])], [np.array([0.0, 1.0])]]
    projected = [np.array([0.0, 0.0])]
    report = verify_kkt(d, projected, gradients, [0, 1], [1.0, 2.0])
    assert report["normalized_primal"] == 0
    with pytest.raises(ValueError, match="primal"):
        verify_kkt(d, [np.array([0.0, 0.1])], gradients, [0, 1], [1.0, 2.0])


def valid_evidence() -> dict[str, Any]:
    steps = [
        {"step": i, "trials": [{"scale": scale} for scale in range(8)]}
        for i in range(1, 17)
    ]
    for step in steps:
        step["solver"] = {
            "kkt_inputs": {
                "displacement": [np.array([1.0])],
                "projected": [np.array([0.0])],
                "gradients": [[np.array([1.0])]],
            },
            "source_indices": [0],
            "dual": [1.0],
        }
    return {
        "schema": "seed457-evidence-v1",
        "decision": {"classification": "close_fresh_subgroup_projection_branch"},
        "arms": {arm: {"steps": copy.deepcopy(steps)} for arm in ("C", "T")},
    }


def test_semantic_evidence_fixture_accepts_valid_record_and_rejects_tamper() -> None:
    evidence = valid_evidence()
    assert (
        verify_evidence_structure(evidence)["status"] == "structure_only_not_verified"
    )
    changed = copy.deepcopy(evidence)
    changed["arms"]["C"]["steps"].pop()
    with pytest.raises(ValueError, match="step_count_invalid:C"):
        verify_evidence_structure(changed)


def test_treatment_evidence_requires_kkt_solver_record() -> None:
    evidence = valid_evidence()
    evidence["arms"]["T"]["steps"][0].pop("solver")
    with pytest.raises(ValueError, match="solver_evidence_missing"):
        verify_evidence_structure(evidence)
