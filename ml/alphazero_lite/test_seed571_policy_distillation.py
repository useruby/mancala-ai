from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from ml.alphazero_lite import seed571_policy_distillation as protocol


def test_input_mismatch_rejected_before_search_or_training(tmp_path, monkeypatch):
    path = tmp_path / "changed-input"
    path.write_bytes(b"changed")
    called = False

    def search(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(protocol.self_play.PUCT, "run", search)
    with pytest.raises(ValueError, match="input_binding_mismatch"):
        protocol.verify_bindings(
            tmp_path, {"changed-input": hashlib.sha256(b"frozen").hexdigest()}
        )
    assert not called


def test_visit_targets_respect_legality_and_zero_unvisited_legal_actions():
    target = protocol.normalized_visit_target([0, 7, 0, 0, 0, 0], [0, 1, 3])
    np.testing.assert_array_equal(target, [0, 1, 0, 0, 0, 0])
    assert target[[2, 4, 5]].sum() == 0
    with pytest.raises(ValueError, match="illegal_action_has_visits"):
        protocol.normalized_visit_target([0, 0, 1, 0, 0, 0], [0, 1])


def test_training_step_changes_only_policy_and_trunk_parameters():
    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Linear(3, 4)
            self.policy_head = nn.Linear(4, 6)
            self.value_head = nn.Linear(4, 1)

        def forward(self, x):
            hidden = torch.tanh(self.trunk(x))
            return self.policy_head(hidden), self.value_head(hidden)

    model = TinyPolicy()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("trunk.", "policy_head.")))
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    x = torch.tensor([[0.2, 0.5, 0.8], [0.7, 0.1, 0.3]], dtype=torch.float32)
    legal = torch.tensor(
        [
            [True, True, False, True, False, False],
            [False, True, True, False, True, False],
        ]
    )
    targets = torch.tensor(
        [[0.25, 0.75, 0, 0, 0, 0], [0, 0, 0.5, 0, 0.5, 0]], dtype=torch.float32
    )
    protocol.train_synthetic_step(model, optimizer, x, targets, legal)
    params = dict(model.named_parameters())
    assert all(torch.equal(params[name], value) for name, value in frozen.items())
    assert all(
        not torch.equal(params[name], value) for name, value in trainable_before.items()
    )


def test_training_order_is_frozen_and_complete():
    order = protocol.fixed_order()
    assert sorted(order) == list(range(256))
    assert order == protocol.fixed_order()


def test_changed_configuration_is_rejected(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"updates": 513}')
    frozen_digest = hashlib.sha256(b'{"updates": 512}').hexdigest()
    with pytest.raises(ValueError, match="configuration_binding_mismatch"):
        protocol.verify_configuration(config, frozen_digest)
