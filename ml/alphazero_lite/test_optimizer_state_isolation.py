"""Regression tests for frozen Adam state handling."""

from __future__ import annotations

import copy
import unittest

import torch

from ml.alphazero_lite.run_fresh_p1_onpolicy_shadow_replay import (
    load_isolated_optimizer,
    optimizer_state_sha256,
)


class OptimizerStateIsolationTest(unittest.TestCase):
    def test_fingerprint_tracks_tensor_and_group_values(self) -> None:
        state = {
            "state": {0: {"step": torch.tensor(1.0), "exp_avg": torch.ones(2)}},
            "param_groups": [{"lr": 1e-5, "params": [0]}],
        }
        original = optimizer_state_sha256(state)
        self.assertEqual(original, optimizer_state_sha256(copy.deepcopy(state)))
        state["state"][0]["exp_avg"][0] = 2.0
        self.assertNotEqual(original, optimizer_state_sha256(state))

    def test_loading_adam_does_not_alias_supplied_state(self) -> None:
        model = torch.nn.Linear(2, 1)
        source = torch.optim.Adam(model.parameters(), lr=1e-5)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        source.step()
        frozen = copy.deepcopy(source.state_dict())
        before = optimizer_state_sha256(frozen)

        isolated = load_isolated_optimizer(model, frozen)
        isolated.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).sum().backward()
        isolated.step()

        self.assertEqual(before, optimizer_state_sha256(frozen))


if __name__ == "__main__":
    unittest.main()
