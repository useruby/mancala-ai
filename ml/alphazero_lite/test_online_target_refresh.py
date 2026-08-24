"""Contracts for pre-update, deterministic online target generation."""

import numpy as np
import torch

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.online_target_refresh import (
    TorchEvaluator,
    model_state_digest,
    target_seed,
)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = torch.nn.Parameter(torch.arange(6, dtype=torch.float32))
        self.value = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.expand(x.shape[0], -1), self.value.expand(x.shape[0], 1)


def test_torch_evaluator_uses_the_live_in_memory_model() -> None:
    model = _Model()
    game = KalahGame([4] * 12, [0, 0], 0)
    evaluator = TorchEvaluator(model, torch.device("cpu"), "kalah_v3")

    before, _ = evaluator.evaluate(game)
    with torch.no_grad():
        model.policy[0] = 100.0
    after, _ = evaluator.evaluate(game)

    assert not np.array_equal(before, after)
    assert int(np.argmax(after)) == 0


def test_pre_update_digest_changes_only_after_the_optimizer_step() -> None:
    model = _Model()
    pre_target_digest = model_state_digest(model)
    frozen_target_digest = pre_target_digest
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    model.policy.sum().backward()
    optimizer.step()

    assert frozen_target_digest == pre_target_digest
    assert model_state_digest(model) != frozen_target_digest


def test_target_seed_is_state_and_type_stable_not_step_dependent() -> None:
    assert target_seed("state", "online_shadow") == target_seed(
        "state", "online_shadow"
    )
    assert target_seed("state", "online_shadow") != target_seed(
        "state", "online_ordinary"
    )
