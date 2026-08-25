"""Invariants for the expanded policy-head parent-Q lane."""

from __future__ import annotations

import torch

from ml.alphazero_lite.root_q_trust_region import (
    adam_proposal_and_restore,
    apply_delta,
)
from ml.alphazero_lite.train import (
    PolicyValueNet,
    apply_trainable_scope,
    input_size_for_encoding,
)


def _model() -> PolicyValueNet:
    torch.manual_seed(235)
    model = PolicyValueNet((96, 3), "residual_v3", input_size_for_encoding("kalah_v3"))
    apply_trainable_scope(model, "policy_head")
    return model


def _step(model: PolicyValueNet, *, line_search: bool) -> dict[str, torch.Tensor]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=1e-5, weight_decay=0.0)
    x = torch.arange(27 * 8, dtype=torch.float32).reshape(8, 27) / 100.0
    target = torch.full((8, 6), 1.0 / 6.0)
    logits, _value = model(x)
    (-(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()).backward()
    if line_search:
        delta = adam_proposal_and_restore(parameters, optimizer)
        apply_delta(parameters, delta, 1.0)
    else:
        optimizer.step()
    return {
        key: value.detach().clone() for key, value in model.state_dict().items()
    } | {
        f"optimizer:{index}:{name}": value.detach().clone()
        for index, state in enumerate(optimizer.state.values())
        for name, value in (
            ("exp_avg", state["exp_avg"]),
            ("exp_avg_sq", state["exp_avg_sq"]),
        )
    }


def test_policy_head_forced_lambda_one_matches_ordinary_adam() -> None:
    ordinary = _step(_model(), line_search=False)
    constrained = _step(_model(), line_search=True)
    assert ordinary.keys() == constrained.keys()
    assert all(torch.equal(ordinary[key], constrained[key]) for key in ordinary)


def test_policy_head_scope_freezes_every_non_policy_tensor() -> None:
    model = _model()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    after = _step(model, line_search=True)
    for key, value in before.items():
        if key.startswith(("policy_hidden_layer.", "policy_head.")):
            continue
        assert torch.equal(value, after[key]), key
