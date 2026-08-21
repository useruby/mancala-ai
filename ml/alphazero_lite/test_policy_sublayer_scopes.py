from __future__ import annotations

import pytest
import torch

from ml.alphazero_lite.train import PolicyValueNet, apply_trainable_scope


def _model(model_type: str = "residual_v3") -> PolicyValueNet:
    return PolicyValueNet((16, 2), model_type, 18)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (
            "policy_hidden_only",
            {"policy_hidden_layer.weight", "policy_hidden_layer.bias"},
        ),
        (
            "policy_readout_only",
            {"policy_head.weight", "policy_head.bias"},
        ),
    ],
)
def test_policy_sublayer_scopes_train_exactly_one_residual_v3_family(
    scope: str, expected: set[str]
) -> None:
    model = _model()

    apply_trainable_scope(model, scope)

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == expected


@pytest.mark.parametrize("scope", ["policy_hidden_only", "policy_readout_only"])
def test_policy_sublayer_scopes_reject_factorized_readout_semantics(scope: str) -> None:
    with pytest.raises(ValueError, match="only supported for residual_v3"):
        apply_trainable_scope(_model("residual_v4_move_factorized"), scope)


def test_policy_sublayer_scopes_preserve_full_policy_gradients() -> None:
    """Freezing a policy sublayer must not alter its policy-loss gradient."""
    torch.manual_seed(7)
    parent = _model()
    states = torch.randn(8, 18)
    targets = torch.randint(0, 6, (8,))
    models = {}
    for scope in ("policy_head", "policy_hidden_only", "policy_readout_only"):
        model = _model()
        model.load_state_dict(parent.state_dict())
        apply_trainable_scope(model, scope)
        logits, _value = model(states)
        torch.nn.functional.cross_entropy(logits, targets).backward()
        models[scope] = model

    full = dict(models["policy_head"].named_parameters())
    hidden = dict(models["policy_hidden_only"].named_parameters())
    readout = dict(models["policy_readout_only"].named_parameters())
    for name in ("policy_hidden_layer.weight", "policy_hidden_layer.bias"):
        assert torch.allclose(hidden[name].grad, full[name].grad)
        assert readout[name].grad is None
    for name in ("policy_head.weight", "policy_head.bias"):
        assert torch.allclose(readout[name].grad, full[name].grad)
        assert hidden[name].grad is None
    for model in models.values():
        for name, parameter in model.named_parameters():
            if name.startswith(
                (
                    "input_layer.",
                    "residual_layers.",
                    "value_hidden_layer.",
                    "value_head.",
                )
            ):
                assert parameter.grad is None
