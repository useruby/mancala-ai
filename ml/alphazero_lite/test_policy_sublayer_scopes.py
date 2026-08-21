from __future__ import annotations

import pytest

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
