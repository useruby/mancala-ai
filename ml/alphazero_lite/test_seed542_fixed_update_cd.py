from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.alphazero_lite import run_seed542_fixed_update_cd as runner, train


def test_registered_schedule_preserves_matched_short_batches_and_order() -> None:
    schedule = runner.load_schedule()
    batches = schedule["updates"]
    assert [item["update"] for item in batches] == list(range(1, 109))
    assert batches[0]["update"] == 1 and batches[-1]["update"] == 108
    assert [item["batch_size"] for item in batches] == [
        n for _ in range(4) for n in ([512] * 26 + [412])
    ]
    assert len(batches) == 108
    assert sum(item["batch_size"] for item in batches) == 54896
    for item in batches:
        assert len(item["C"]) == item["batch_size"]
        assert len(item["D"]) == item["batch_size"]
        assert len(set(item["D"])) == len(item["D"])


def test_real_model_update_reports_weighted_losses_clipping_and_adam_state() -> None:
    source = runner.CORPUS / "shard-542000.jsonl"
    x, p, v, weights = train.load_jsonl(source, include_policy_loss_weights=True)
    state = x[0]
    legal = train.derive_legal_moves_from_encoded_state(state)
    assert legal
    first = np.zeros(6, dtype=np.float32)
    second = np.zeros(6, dtype=np.float32)
    first[legal[0]] = 1.0
    second[legal[-1]] = 1.0
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, runner.INITIALIZER)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    ids = ["row-a", "row-b"]
    positions = {"row-a": 0, "row-b": 1}
    trace = runner.update(
        model,
        optimizer,
        ids,
        positions,
        np.asarray([state, state], dtype=np.float32),
        np.asarray([first, second], dtype=np.float32),
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )
    assert trace["row_ids"] == ids
    assert trace["batch_size"] == 2
    assert trace["optimizer_step"] == 1
    assert trace["gradient_norm_preclip"] > 1.0
    assert trace["weighted_total_objective"] == pytest.approx(
        trace["policy_loss"] + 0.5 * trace["value_loss"], abs=2e-7
    )
    assert len(trace["parameter_state_sha256"]) == 64


def test_schedule_loader_rejects_altered_published_batch_lengths(monkeypatch) -> None:
    schedule = runner.load_schedule()
    schedule["updates"][-1]["batch_size"] += 1
    original = runner.PROTOCOL
    path = original.with_name("invalid-test-protocol.json")
    try:
        path.write_text(__import__("json").dumps(schedule))
        monkeypatch.setattr(runner, "PROTOCOL", path)
        with pytest.raises(ValueError, match="published_schedule_shape_mismatch"):
            runner.load_schedule()
    finally:
        path.unlink(missing_ok=True)
