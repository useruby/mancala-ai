"""Synthetic contract tests for the separate probe-reuse generation harness."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ml.alphazero_lite import validate_varied_opponent_probe_reuse as validation


def test_freeze_is_exact_sixteen_pair_subset_of_580() -> None:
    freeze = validation.build_freeze()
    jobs = freeze["pairs"]
    assert len(jobs) == 16
    assert sum(job["game_specs"][0]["arm"] == "control" for job in jobs) == 8
    assert (
        sum(job["game_specs"][0].get("opponent") == "initializer" for job in jobs) == 4
    )
    assert sum(job["game_specs"][0].get("opponent") == "T512" for job in jobs) == 4
    original = validation.json.loads(
        (validation.ROOT / validation.BASE_FREEZE).read_text()
    )
    original_specs = {spec["game_id"]: spec for spec in original["assignments"]}
    for job in jobs:
        for spec in job["game_specs"]:
            assert spec == original_specs[spec["game_id"]]


def test_execution_plan_runs_pairs_back_to_back_and_alternates_first() -> None:
    jobs = validation.build_freeze()["pairs"][:4]
    plan = validation.execution_plan(jobs)
    assert [(item[0], item[1]) for item in plan] == [
        (jobs[0], "v1"),
        (jobs[0], "v2"),
        (jobs[1], "v2"),
        (jobs[1], "v1"),
        (jobs[2], "v1"),
        (jobs[2], "v2"),
        (jobs[3], "v2"),
        (jobs[3], "v1"),
    ]


def test_resume_returns_committed_result_without_dispatching_worker(
    tmp_path: Path,
) -> None:
    job = {"game_specs": [{"pair_id": "control-opening-0000", "designated_p_seat": 0}]}
    result = {"pair_id": "control-opening-0000", "games": []}
    with (
        patch.object(
            validation.pilot, "load_pair_checkpoints", return_value=([result], [])
        ),
        patch.object(validation.pilot, "reconcile_pair_attempts", return_value={}),
        patch.object(validation.pilot, "pair_attempt_path") as attempt_path,
        patch.object(validation.concurrent.futures, "ProcessPoolExecutor") as executor,
    ):
        resumed = validation._run_one(
            "v1", job, tmp_path, tmp_path, {"freeze_sha256": "f"}
        )
    assert resumed is result
    attempt_path.assert_not_called()
    executor.assert_not_called()


def test_semantic_comparison_catches_trajectory_and_retained_row_mismatch() -> None:
    game = {
        "spec": {"game_id": "game-1"},
        "trajectory": [1, 2, 3],
        "winner": 0,
        "rows": [{"row_id": "row-1", "value": 1.0}],
        "cpu_time_ns": 12,
        "search_settings": {"simulations": 16},
    }
    equal = {**game, "cpu_time_ns": 900}
    assert validation.verify_output_pair({"games": [game]}, {"games": [equal]}) == []
    changed = {**equal, "rows": [{"row_id": "row-1", "value": 0.5}]}
    assert validation.verify_output_pair({"games": [game]}, {"games": [changed]})
    changed_settings = {**equal, "search_settings": {"simulations": 32}}
    assert validation.verify_output_pair(
        {"games": [game]}, {"games": [changed_settings]}
    )


def test_historical_snapshot_is_not_accepted_by_live_freeze_check() -> None:
    freeze_path = validation.ROOT / validation.OUTPUT_ROOT / "freeze.json"
    freeze = validation.json.loads(freeze_path.read_text())
    with pytest.raises(ValueError, match="validation_source_hash_mismatch"):
        validation.verify_freeze(freeze)
    with pytest.raises(ValueError, match="validation_source_hash_mismatch"):
        validation.preflight(freeze, Path("unused-artifacts"))
    with pytest.raises(ValueError, match="validation_source_hash_mismatch"):
        validation.execute(
            freeze_path,
            Path("unused-artifacts"),
            Path("unused-output"),
            freeze["freeze_sha256"],
        )
    validation.verify_freeze(freeze, allow_historical_snapshot=True)
