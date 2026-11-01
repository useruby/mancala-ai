"""Self-contained tests for seed549 freezes, recovery, and derivatives."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.alphazero_lite import seed549_corpus as corpus


def test_generation_command_freezes_all_pilot_worker_settings() -> None:
    command = corpus.make_command(
        corpus.ROOT, "control", 6, corpus.OUT / "control/shard-006.jsonl"
    )
    assert command[command.index("--games") + 1] == "64"
    assert command[command.index("--seed") + 1] == "549006"
    assert command[command.index("--simulations") + 1] == "128"
    assert command[command.index("--workers") + 1] == "1"
    assert command[command.index("--max-moves") + 1] == "200"
    assert command[command.index("--start-state-mode") + 1] == "standard_4x6"
    assert command[command.index("--policy-target-noise-mode") + 1] == "denoised"
    assert command[command.index("--value-target-mode") + 1] == "default"


def test_partition_is_frozen_before_generation() -> None:
    assert corpus.TRAIN_SHARDS == {0, 1, 2, 3, 4, 5}
    assert set(range(corpus.SHARDS)) - corpus.TRAIN_SHARDS == {6, 7}
    assert len(corpus.TRAIN_SHARDS) * corpus.GAMES_PER_SHARD == 384
    assert 2 * corpus.GAMES_PER_SHARD == 128


def test_schedule_has_108_updates_54896_exposures_and_pass_uniqueness() -> None:
    schedule = corpus.build_schedule(701, seed=17)
    assert schedule["updates"] == 108
    assert schedule["exposures"] == 54_896
    batches = schedule["batches"]
    assert len(batches) == 108
    assert [
        b["end_exposure_exclusive"] - b["start_exposure"] for b in batches
    ] == corpus.BATCHES * 4
    flattened = [
        index for batch in batches for index in batch["row_occurrence_indices"]
    ]
    assert len(flattened) == 54_896
    for boundary in schedule["pass_boundaries"]:
        values = flattened[
            boundary["start_exposure"] : boundary["end_exposure_exclusive"]
        ]
        assert len(values) == len(set(values))
    assert schedule == corpus.build_schedule(701, seed=17)


def test_schedule_preserves_duplicate_row_occurrences() -> None:
    # The schedule indexes source row occurrences; identical rows are not deduplicated.
    schedule = corpus.build_schedule(2, seed=29)
    assert schedule["exposures"] == 54_896
    first_pass = schedule["pass_boundaries"][0]
    values = [
        i for batch in schedule["batches"] for i in batch["row_occurrence_indices"]
    ]
    pass_values = values[
        first_pass["start_exposure"] : first_pass["end_exposure_exclusive"]
    ]
    assert set(pass_values) == {0, 1}
    assert len(pass_values) == 2


def test_development_exclusions_preserve_identity_reason_and_source() -> None:
    training = {"control": {"canonical-a", "float-b"}, "treatment": {"float-c"}}
    assert corpus.development_exclusion_reasons("canonical-a", "float-c", training) == [
        "canonical_overlap:control",
        "float32_overlap:treatment",
    ]
    assert corpus.development_exclusion_reasons("new", "new-float", training) == []


def test_identity_mutation_changes_float32_identity() -> None:
    a = {"_canonical_state": "[1,2]", "state": [0.1, 0.2]}
    b = {"_canonical_state": "[1,2]", "state": [0.1, 0.3]}
    assert corpus.identity(a)[0] == corpus.identity(b)[0]
    assert corpus.identity(a)[1] != corpus.identity(b)[1]


def test_worker_rejects_mutated_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(corpus, "preflight", lambda: {})
    with pytest.raises(ValueError, match="worker_command_mismatch"):
        corpus.run_worker(
            "control", ["--arm", "control", "--shard", "0", "--out", "bad.jsonl"]
        )


def test_resume_skips_only_receipt_verified_completed_shard(tmp_path) -> None:
    output = tmp_path / "shard.jsonl"
    output.write_text('{"row":1}\n')
    receipt = corpus.digest(output)
    events = [
        {"event": "started"},
        {"event": "finished", "output_sha256": receipt},
        {"event": "published", "output_sha256": receipt},
    ]
    assert corpus.completed_receipt_verified(events, output)
    output.write_text('{"row":2}\n')
    assert not corpus.completed_receipt_verified(events, output)
    assert not corpus.completed_receipt_verified(events[:2], output)
    assert not corpus.completed_receipt_verified(
        [events[0], {"event": "interrupted"}], output
    )


def test_mutated_registration_freeze_is_rejected(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A changed registration byte without a matching freeze is rejected before assets are read.
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"generation": {}}))
    (tmp_path / "execution-freeze.json").write_text(
        json.dumps({"registration_sha256": "bad"})
    )
    monkeypatch.setattr(
        corpus,
        "OUT",
        tmp_path.relative_to(corpus.ROOT)
        if tmp_path.is_relative_to(corpus.ROOT)
        else corpus.OUT,
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        corpus.preflight(tmp_path)


def test_passes_are_deterministic_and_do_not_repeat_within_each_pass() -> None:
    schedule = corpus.build_schedule(997, seed=549000)
    flat = [
        value
        for batch in schedule["batches"]
        for value in batch["row_occurrence_indices"]
    ]
    boundaries = schedule["pass_boundaries"]
    assert boundaries[0]["permutation_seed"] == 549000
    assert boundaries[1]["permutation_seed"] == 549001
    for item in boundaries:
        span = flat[item["start_exposure"] : item["end_exposure_exclusive"]]
        assert len(span) == len(set(span))
    assert np.array_equal(
        np.asarray(flat),
        np.asarray(
            [
                value
                for batch in corpus.build_schedule(997, seed=549000)["batches"]
                for value in batch["row_occurrence_indices"]
            ]
        ),
    )
