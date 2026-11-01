from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from ml.alphazero_lite import (
    run_seed542_fixed_update_cd as fixed_update,
    run_seed542_fixed_update_cd_v2 as v2,
    seed542_training_budget_432 as experiment,
    train,
    verify_seed542_training_budget_432 as verifier,
)


def test_full_schedule_finishes_unused_rows_before_seeded_pass() -> None:
    _x, _p, _v, _weights, identities = fixed_update.load_sources("D")
    schedule = experiment.build_schedule({}, identities)
    assert len(schedule) == 432
    assert [item["update"] for item in schedule] == list(range(1, 433))
    assert [item["phase"] for item in schedule[:108]] == [
        "registered_initial_schedule"
    ] * 108
    assert [item["batch_size"] for item in schedule[108:215]] == [512] * 106 + [336]
    assert [item["phase"] for item in schedule[108:215]] == [
        "finish_first_permutation"
    ] * 107
    assert all(
        row["phase"] == "deterministic_shuffled_passes" for row in schedule[215:]
    )
    assert schedule[215]["shuffle_seeds"] == [543]
    assert schedule[-1]["shuffle_seeds"] == [544]
    assert sum(row["batch_size"] for row in schedule) == 220608
    first_rows = [row for batch in schedule[:108] for row in batch["rows"]]
    remainder = [row for batch in schedule[108:215] for row in batch["rows"]]
    assert len(set(first_rows)) == len(first_rows) == 54896
    assert len(set(remainder)) == len(remainder) == 54608
    assert not set(first_rows) & set(remainder)


def test_synthetic_continuation_keeps_same_adam_state() -> None:
    source = fixed_update.CORPUS / "shard-542000.jsonl"
    states, _policies, _values, _weights = train.load_jsonl(
        source, include_policy_loss_weights=True
    )
    state = states[0]
    legal = train.derive_legal_moves_from_encoded_state(state)
    x = np.asarray([state, state], dtype=np.float32)
    policy = np.zeros((2, 6), dtype=np.float32)
    policy[0, legal[0]] = 1.0
    policy[1, legal[-1]] = 1.0
    value = np.asarray([[0.0], [1.0]], dtype=np.float32)
    weights = np.ones(2, dtype=np.float32)
    identities = ["synthetic:0", "synthetic:1"]
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, fixed_update.INITIALIZER)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    recorder = v2.DirectRecorder(model, x, policy, value, weights, identities)
    positions = {identity: index for index, identity in enumerate(identities)}
    optimizer_identity = id(optimizer)
    first = experiment.apply_schedule_item(
        recorder,
        optimizer,
        {"update": 108, "phase": "control", "rows": identities},
        positions,
    )
    second = experiment.apply_schedule_item(
        recorder,
        optimizer,
        {"update": 109, "phase": "continuation", "rows": identities[::-1]},
        positions,
    )
    recorder.close()
    assert id(optimizer) == optimizer_identity
    assert first["optimizer_step"] == 1
    assert second["optimizer_step"] == 2
    assert all(int(state["step"].item()) == 2 for state in optimizer.state.values())


def test_control_mismatch_is_identified_and_stops_control() -> None:
    expected = {"update": 1, "parameter_state_after": "good"}
    observed = {"update": 1, "parameter_state_after": "bad"}
    assert verifier.control_mismatches(expected, observed) == ["parameter_state_after"]


def test_schedule_tampering_is_detected() -> None:
    contents = b'[{"update":1}]\n'
    expected = hashlib.sha256(contents).hexdigest()
    experiment.validate_schedule_integrity(contents, expected)
    with pytest.raises(ValueError, match="432_frozen_schedule_changed"):
        experiment.validate_schedule_integrity(contents + b" ", expected)
