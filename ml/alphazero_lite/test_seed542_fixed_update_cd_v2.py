from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite import (
    run_seed542_fixed_update_cd as fixed_update,
    run_seed542_fixed_update_cd_v2 as v2,
    train,
    verify_seed542_fixed_update_cd_v2 as verifier,
)


def test_published_schedule_validates_one_based_arms_and_short_batches() -> None:
    protocol = json.loads(v2.CORPUS.joinpath("training-protocol.json").read_text())
    schedule = verifier.published_schedule(protocol)
    assert [item["update"] for item in schedule] == list(range(1, 109))
    assert [item["batch_size"] for item in schedule] == [
        size for _ in range(4) for size in ([512] * 26 + [412])
    ]
    assert sum(item["batch_size"] for item in schedule) == 54896
    altered = copy.deepcopy(protocol)
    altered["expanded"]["schedule"][27]["rows"][0] = "altered-row"
    with pytest.raises(ValueError, match="v2_published_schedule_view_mismatch"):
        verifier.published_schedule(altered)


def test_production_and_direct_update_match_full_trace_and_adam_state() -> None:
    source = fixed_update.CORPUS / "shard-542000.jsonl"
    states, _policies, _values, _weights = train.load_jsonl(
        source, include_policy_loss_weights=True
    )
    state = states[0]
    legal = train.derive_legal_moves_from_encoded_state(state)
    assert legal
    target_a = np.zeros(6, dtype=np.float32)
    target_b = np.zeros(6, dtype=np.float32)
    target_a[legal[0]] = 1.0
    target_b[legal[-1]] = 1.0
    x = np.asarray([state, state], dtype=np.float32)
    p = np.asarray([target_a, target_b], dtype=np.float32)
    value = np.asarray([[0.0], [1.0]], dtype=np.float32)
    weights = np.ones(2, dtype=np.float32)
    identities = ["synthetic:1", "synthetic:2"]

    production_model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(production_model, fixed_update.INITIALIZER)
    direct_model = copy.deepcopy(production_model)
    production_optimizer = torch.optim.Adam(
        production_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    direct_optimizer = torch.optim.Adam(
        direct_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    recorder = v2.ProductionRecorder(identities, (x, p, value, weights))
    recorder.attach(production_model)
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, max_norm, *args, **kwargs):
        result = original_clip(parameters, max_norm, *args, **kwargs)
        recorder.latest_clip_return = float(result.detach().cpu().item())
        return result

    torch.nn.utils.clip_grad_norm_ = capture_clip
    generator = torch.Generator(device="cpu").manual_seed(504)
    try:
        train.train_one_epoch(
            model=production_model,
            optimizer=production_optimizer,
            compact_x=x,
            compact_p=p,
            compact_v=value,
            compact_policy_loss_weights=weights,
            replay_indexes=np.arange(2, dtype=np.int64),
            batch_size=2,
            device=torch.device("cpu"),
            value_loss_weight=0.5,
            value_loss="huber",
            huber_delta=1.0,
            grad_clip=1.0,
            primary_order_generator=generator,
            step_callback=recorder.step_callback,
            step_callback_needs_raw_gradients=True,
            loss_observer=recorder.loss_observer,
        )
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        recorder.close()
    order = torch.randperm(
        2, generator=torch.Generator(device="cpu").manual_seed(504)
    ).tolist()
    direct = v2.DirectRecorder(direct_model, x, p, value, weights, identities)
    actual = direct.update(
        direct_optimizer,
        [identities[index] for index in order],
        {name: index for index, name in enumerate(identities)},
    )
    direct.close()
    actual["update"] = 1
    production = recorder.records[0]
    assert verifier.validate_record_pair(production, actual) == []
    assert production["batch"]["row_ids"] == actual["batch"]["row_ids"]
    assert (
        production["batch"]["tensors"]["target_policy"]["sha256"]
        == actual["batch"]["tensors"]["target_policy"]["sha256"]
    )
    assert (
        production["batch"]["tensors"]["input"]["sha256"]
        == actual["batch"]["tensors"]["input"]["sha256"]
    )


def test_record_checkpoint_prediction_and_metric_tampering_is_detected() -> None:
    production = {"update": 1, "policy_loss": 0.5, "parameter_state_after": "abc"}
    direct = dict(production)
    assert verifier.validate_record_pair(production, direct) == []
    direct["policy_loss"] = np.nextafter(0.5, 1.0).item()
    assert "policy_loss" in verifier.validate_record_pair(production, direct)
    assert verifier.compare_checkpoint_arrays(
        {"x": np.asarray([1.0], dtype=np.float32)},
        {"x": np.asarray([1.0], dtype=np.float32)},
    )
    assert not verifier.compare_checkpoint_arrays(
        {"x": np.asarray([1.0], dtype=np.float32)},
        {
            "x": np.asarray(
                [np.nextafter(np.float32(1), np.float32(2))], dtype=np.float32
            )
        },
    )
    expected = {"predictions": [{"row_id": "a", "value": 1.0}], "effect": -0.1}
    verifier.assert_json_equal(expected, copy.deepcopy(expected), "unused")
    altered = copy.deepcopy(expected)
    altered["predictions"][0]["value"] = 0.0
    with pytest.raises(ValueError, match="tampered"):
        verifier.assert_json_equal(expected, altered, "tampered")


def test_read_only_comparison_does_not_modify_source_bytes_or_mtime(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "registered.json"
    evidence.write_text('{"evidence":true}\n')
    before = (evidence.read_bytes(), evidence.stat().st_mtime_ns)
    verifier.assert_json_equal(
        {"evidence": True}, json.loads(evidence.read_text()), "mismatch"
    )
    with pytest.raises(ValueError):
        verifier.assert_json_equal(
            {"evidence": False}, json.loads(evidence.read_text()), "tampered"
        )
    assert evidence.read_bytes() == before[0]
    assert evidence.stat().st_mtime_ns == before[1]
