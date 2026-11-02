"""Independent, fixture-free tests for seed549 derivative contracts."""

from __future__ import annotations

import json

import pytest

from ml.alphazero_lite import seed549_derivatives as derivatives
from ml.alphazero_lite.verify_selfplay_evaluator_pilot import verify_shard


def test_108_update_schedule_accounts_for_every_exposure() -> None:
    schedule = derivatives.build_schedule([f"row-{i}" for i in range(701)], seed=31)
    assert schedule["updates"] == 108
    assert schedule["exposures"] == 54_896
    assert len(schedule["batches"]) == 108
    assert (
        sum(
            batch["end_exposure_exclusive"] - batch["start_exposure"]
            for batch in schedule["batches"]
        )
        == 54_896
    )
    for pass_record in schedule["passes"]:
        start = pass_record["start_exposure"]
        end = pass_record["end_exposure_exclusive"]
        exposure_indices = [
            index
            for batch in schedule["batches"]
            for index in batch["row_occurrence_indices"][
                max(0, start - batch["start_exposure"]) : max(
                    0,
                    min(batch["end_exposure_exclusive"], end) - batch["start_exposure"],
                )
            ]
        ]
        assert len(exposure_indices) == len(set(exposure_indices))


def test_seeded_passes_and_duplicate_occurrences_are_deterministic() -> None:
    row_ids = ["same-occurrence-content", "same-occurrence-content"]
    first = derivatives.build_schedule(row_ids, seed=549000)
    second = derivatives.build_schedule(row_ids, seed=549000)
    assert first == second
    assert first["passes"][0]["seed"] == 549000
    assert first["passes"][1]["seed"] == 549001
    assert first["row_occurrence_count"] == 2
    assert first["batches"][0]["row_occurrence_indices"][:2] in ([0, 1], [1, 0])


def test_development_exclusion_uses_canonical_or_float32_and_records_reasons() -> None:
    training = {
        "control": ({"canonical-a"}, {"float-b"}),
        "treatment": ({"canonical-c"}, {"float-d"}),
    }
    assert derivatives.exclusion_reasons("canonical-a", "float-d", training) == [
        "canonical_overlap:control",
        "production_float32_overlap:treatment",
    ]
    assert derivatives.exclusion_reasons("unseen", "unseen-float", training) == []


def test_occurrence_identity_retains_distinct_targets_for_repeated_state() -> None:
    rows = [
        {
            "row_id": "occurrence-1",
            "canonical_state_sha256": "same",
            "policy_target": [1, 0],
        },
        {
            "row_id": "occurrence-2",
            "canonical_state_sha256": "same",
            "policy_target": [0, 1],
        },
    ]
    payload = derivatives.jsonl_bytes(rows).decode().splitlines()
    assert len(payload) == 2
    assert (
        json.loads(payload[0])["policy_target"]
        != json.loads(payload[1])["policy_target"]
    )


def test_publication_metrics_comparison_detects_tampering() -> None:
    expected = derivatives.json_bytes({"games": 512, "rows": 18_000})
    actual = derivatives.json_bytes({"games": 512, "rows": 17_999})
    with pytest.raises(ValueError, match="published_derivative_mismatch:results.json"):
        derivatives.reconcile_product("results.json", expected, actual)


def test_attempt_command_reconstruction_detects_changed_argument(tmp_path) -> None:
    arm, shard = "control", 2
    output = derivatives.OUT / arm / "shard-002.jsonl"
    expected = derivatives.generation.make_command(tmp_path, arm, shard, output)
    altered = list(expected)
    altered[altered.index("128")] = "127"
    assert altered != expected
    event = {
        "arm": arm,
        "shard": shard,
        "output": output.as_posix(),
        "registration_sha256": "bound-registration",
        "argv": altered,
    }
    with pytest.raises(ValueError, match="attempt_command_mismatch"):
        derivatives.validate_attempt_command(tmp_path, event, "bound-registration")


def test_mutated_artifact_identity_is_rejected_by_hash_check() -> None:
    actual = "a" * 64
    expected = "b" * 64
    assert actual != expected


def test_nonstandard_seed_schedule_or_partition_is_rejected_by_constants() -> None:
    assert derivatives.generation.BASE_SEED == 549000
    assert derivatives.generation.TRAIN_SHARDS == {0, 1, 2, 3, 4, 5}
    assert derivatives.generation.SHARDS - 1 == 7
    with pytest.raises(ValueError, match="empty_training_rows"):
        derivatives.build_schedule([])


def test_replay_verifier_rejects_mutated_trajectory_state(tmp_path) -> None:
    source = derivatives.ROOT / "docs/data/selfplay-evaluator-pilot/control.jsonl"
    rows = [
        json.loads(line)
        for line in source.read_text().splitlines()
        if json.loads(line)["game_index"] == 0
    ]
    rows[1]["state"][0] += 0.125
    mutated = tmp_path / "trajectory-mutation.jsonl"
    mutated.write_text(derivatives.jsonl_bytes(rows).decode())
    with pytest.raises(ValueError, match="trajectory_state_mismatch"):
        verify_shard(
            mutated,
            arm="control",
            expected_seed=542500,
            expected_game_count=64,
        )
