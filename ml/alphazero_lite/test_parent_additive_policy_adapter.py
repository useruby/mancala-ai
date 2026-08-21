from __future__ import annotations

import numpy as np
import torch

from ml.alphazero_lite.train import PolicyValueNet, apply_trainable_scope


def test_zero_initialized_adapter_preserves_parent_logits() -> None:
    torch.manual_seed(7)
    parent = PolicyValueNet((16, 2), "residual_v3", 18)
    adapter = PolicyValueNet((16, 2), "residual_v3_parent_additive_policy_adapter", 18)
    adapter.load_state_dict(parent.state_dict(), strict=False)
    x = torch.from_numpy(
        np.random.default_rng(3).normal(size=(4, 18)).astype(np.float32)
    )

    parent_logits, parent_value = parent(x)
    adapter_logits, adapter_value = adapter(x)

    assert torch.equal(parent_logits, adapter_logits)
    assert torch.equal(parent_value, adapter_value)


def test_adapter_scope_trains_only_residual_logits() -> None:
    model = PolicyValueNet((16, 2), "residual_v3_parent_additive_policy_adapter", 18)
    apply_trainable_scope(model, "policy_adapter_only")

    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == {
        "policy_adapter.weight",
        "policy_adapter.bias",
    }
