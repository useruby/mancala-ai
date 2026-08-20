from __future__ import annotations

import torch

from ml.alphazero_lite.policy_sublayer_graft import (
    HIDDEN_KEYS,
    POLICY_KEYS,
    READOUT_KEYS,
    assert_candidate_contract,
    assert_graft_contract,
    graft_state,
)


def _state() -> dict[str, torch.Tensor]:
    return {
        "input_layer.weight": torch.randn(4, 3),
        "residual_layers.0.0.weight": torch.randn(4, 4),
        "policy_hidden_layer.weight": torch.randn(4, 4),
        "policy_hidden_layer.bias": torch.randn(4),
        "policy_head.weight": torch.randn(6, 4),
        "policy_head.bias": torch.randn(6),
        "value_hidden_layer.weight": torch.randn(2, 4),
        "value_head.weight": torch.randn(1, 2),
    }


def test_exact_policy_sublayer_graft_composition() -> None:
    parent = _state()
    candidate = {name: value.clone() for name, value in parent.items()}
    for name in POLICY_KEYS:
        candidate[name] += 0.1

    assert_candidate_contract(parent, candidate)
    hidden = graft_state(parent, candidate, HIDDEN_KEYS)
    readout = graft_state(parent, candidate, READOUT_KEYS)
    full = graft_state(parent, candidate, POLICY_KEYS)

    assert_graft_contract("hidden_delta_only", parent, candidate, hidden)
    assert_graft_contract("readout_delta_only", parent, candidate, readout)
    assert_graft_contract("full", parent, candidate, full)
    assert all(torch.equal(full[name], candidate[name]) for name in full)


def test_candidate_contract_rejects_protected_change() -> None:
    parent, candidate = _state(), _state()
    candidate = {name: value.clone() for name, value in parent.items()}
    candidate["input_layer.weight"] += 0.1
    try:
        assert_candidate_contract(parent, candidate)
    except AssertionError as error:
        assert "non-policy" in str(error)
    else:
        raise AssertionError("protected trunk update was accepted")
