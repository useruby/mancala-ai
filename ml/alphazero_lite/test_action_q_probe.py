from __future__ import annotations

import torch

from ml.alphazero_lite.action_q_probe import ActionQProbe
from ml.alphazero_lite.train import PolicyValueNet, input_size_for_encoding


def test_trunk_extraction_preserves_residual_v3_forward_outputs() -> None:
    torch.manual_seed(236)
    model = PolicyValueNet((96, 3), "residual_v3", input_size_for_encoding("kalah_v3"))
    x = torch.randn(7, input_size_for_encoding("kalah_v3"))
    with torch.no_grad():
        logits, value = model(x)
        h = model.trunk_features(x)
        expected_logits = model.policy_head(torch.relu(model.policy_hidden_layer(h)))
        expected_value = torch.tanh(
            model.value_head(torch.relu(model.value_hidden_layer(h)))
        )
    assert torch.equal(logits, expected_logits)
    assert torch.equal(value, expected_value)


def test_action_q_probe_emits_six_actions() -> None:
    probe = ActionQProbe(96)
    assert probe(torch.zeros(4, 96)).shape == (4, 6)
