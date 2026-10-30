from __future__ import annotations

import copy

import numpy as np
import torch

from ml.alphazero_lite import run_seed542_fixed_update_cd as cd, train


def test_production_and_fixed_c_batch_paths_match_parameters_and_adam() -> None:
    source = cd.CORPUS / "shard-542000.jsonl"
    states, _policies, _values, _weights = train.load_jsonl(
        source, include_policy_loss_weights=True
    )
    state = states[0]
    legal = train.derive_legal_moves_from_encoded_state(state)
    assert legal
    first = np.zeros(6, dtype=np.float32)
    second = np.zeros(6, dtype=np.float32)
    first[legal[0]] = 1.0
    second[legal[-1]] = 1.0
    x = np.asarray([state, state], dtype=np.float32)
    p = np.asarray([first, second], dtype=np.float32)
    v = np.asarray([[0.0], [1.0]], dtype=np.float32)
    weights = np.ones(2, dtype=np.float32)
    row_ids = ["synthetic-source:1", "synthetic-source:2"]

    production_model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(production_model, cd.INITIALIZER)
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
    generator = torch.Generator(device="cpu").manual_seed(542)
    order = torch.randperm(2, generator=generator).tolist()
    production_metrics = train.train_one_epoch(
        model=production_model,
        optimizer=production_optimizer,
        compact_x=x,
        compact_p=p,
        compact_v=v,
        compact_policy_loss_weights=weights,
        replay_indexes=np.arange(2, dtype=np.int64),
        batch_size=2,
        device=torch.device("cpu"),
        value_loss_weight=0.5,
        value_loss="huber",
        huber_delta=1.0,
        grad_clip=1.0,
        primary_order_generator=torch.Generator(device="cpu").manual_seed(542),
    )
    identities = [row_ids[index] for index in order]
    direct_metrics = cd.update(
        direct_model,
        direct_optimizer,
        identities,
        {row_id: index for index, row_id in enumerate(row_ids)},
        x,
        p,
        v,
        weights,
    )
    production_checkpoint = train.checkpoint_from_model(production_model)
    direct_checkpoint = train.checkpoint_from_model(direct_model)
    assert set(production_checkpoint) == set(direct_checkpoint)
    for name in production_checkpoint:
        np.testing.assert_array_equal(
            production_checkpoint[name], direct_checkpoint[name]
        )
    assert production_metrics["policy_loss"] == direct_metrics["policy_loss"]
    assert production_metrics["value_loss"] == direct_metrics["value_loss"]
    assert (
        production_metrics["total_loss"] == direct_metrics["weighted_total_objective"]
    )
    for (production_name, production_parameter), (direct_name, direct_parameter) in zip(
        production_model.named_parameters(),
        direct_model.named_parameters(),
        strict=True,
    ):
        assert production_name == direct_name
        production_state = production_optimizer.state[production_parameter]
        direct_state = direct_optimizer.state[direct_parameter]
        assert production_state.keys() == direct_state.keys()
        for key in production_state:
            left, right = production_state[key], direct_state[key]
            if isinstance(left, torch.Tensor):
                assert torch.equal(left, right), (production_name, key)
            else:
                assert left == right
