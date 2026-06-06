import unittest

import torch

from ml.alphazero_lite.train import (
    SUPPORTED_TRAINABLE_SCOPES,
    PolicyValueNet,
    _count_parameters,
    apply_trainable_scope,
)


class TrainableScopeTest(unittest.TestCase):
    def setUp(self):
        self.model = PolicyValueNet(
            hidden_sizes=(96, 3),
            model_type="residual_v3",
            input_size=27,
        )

    def _collect_grad_states(self) -> dict[str, bool]:
        return {
            name: bool(param.requires_grad)
            for name, param in self.model.named_parameters()
        }

    def _trainable_param_names(self) -> set[str]:
        return {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }

    def test_all_scope_preserves_existing_behaviour(self):
        apply_trainable_scope(self.model, "all")
        total, trainable = _count_parameters(self.model)
        self.assertEqual(total, trainable)
        self.assertGreater(total, 0)
        states = self._collect_grad_states()
        self.assertTrue(all(states.values()), "all params must be trainable")

    def test_policy_head_scope_only_trains_policy_head_params(self):
        apply_trainable_scope(self.model, "policy_head")
        trainable = self._trainable_param_names()
        self.assertIn("policy_hidden_layer.weight", trainable)
        self.assertIn("policy_hidden_layer.bias", trainable)
        self.assertIn("policy_head.weight", trainable)
        self.assertIn("policy_head.bias", trainable)

        frozen = {
            name
            for name, param in self.model.named_parameters()
            if not param.requires_grad
        }
        self.assertIn("input_layer.weight", frozen)
        self.assertIn("input_layer.bias", frozen)
        self.assertIn("value_hidden_layer.weight", frozen)
        self.assertIn("value_hidden_layer.bias", frozen)
        self.assertIn("value_head.weight", frozen)
        self.assertIn("value_head.bias", frozen)
        for block_idx in range(len(self.model.residual_layers)):
            self.assertIn(f"residual_layers.{block_idx}.0.weight", frozen)
            self.assertIn(f"residual_layers.{block_idx}.0.bias", frozen)
            self.assertIn(f"residual_layers.{block_idx}.1.weight", frozen)
            self.assertIn(f"residual_layers.{block_idx}.1.bias", frozen)

    def test_last_block_policy_scope_trains_final_block_and_policy(self):
        apply_trainable_scope(self.model, "last_block_policy")
        trainable = self._trainable_param_names()
        self.assertIn("policy_hidden_layer.weight", trainable)
        self.assertIn("policy_hidden_layer.bias", trainable)
        self.assertIn("policy_head.weight", trainable)
        self.assertIn("policy_head.bias", trainable)

        block_count = len(self.model.residual_layers)
        last_idx = block_count - 1
        self.assertIn(f"residual_layers.{last_idx}.0.weight", trainable)
        self.assertIn(f"residual_layers.{last_idx}.0.bias", trainable)
        self.assertIn(f"residual_layers.{last_idx}.1.weight", trainable)
        self.assertIn(f"residual_layers.{last_idx}.1.bias", trainable)

        frozen = {
            name
            for name, param in self.model.named_parameters()
            if not param.requires_grad
        }
        self.assertIn("input_layer.weight", frozen)
        self.assertIn("input_layer.bias", frozen)
        self.assertIn("value_hidden_layer.weight", frozen)
        self.assertIn("value_hidden_layer.bias", frozen)
        self.assertIn("value_head.weight", frozen)
        self.assertIn("value_head.bias", frozen)
        for block_idx in range(block_count - 1):
            self.assertIn(f"residual_layers.{block_idx}.0.weight", frozen)
            self.assertIn(f"residual_layers.{block_idx}.0.bias", frozen)
            self.assertIn(f"residual_layers.{block_idx}.1.weight", frozen)
            self.assertIn(f"residual_layers.{block_idx}.1.bias", frozen)

    def test_policy_head_scope_parameter_count_matches_expected(self):
        apply_trainable_scope(self.model, "policy_head")
        total, trainable = _count_parameters(self.model)
        expected_policy_head = 96 * 96 + 96 + 96 * 6 + 6  # 9894
        self.assertEqual(trainable, expected_policy_head)
        self.assertGreater(total, trainable)

    def test_last_block_policy_scope_parameter_count_matches_expected(self):
        apply_trainable_scope(self.model, "last_block_policy")
        total, trainable = _count_parameters(self.model)
        expected = (
            96 * 96
            + 96  # first layer of last block
            + 96 * 96
            + 96  # second layer of last block
            + 96 * 96
            + 96  # policy_hidden
            + 96 * 6
            + 6  # policy_head
        )  # 28518
        self.assertEqual(trainable, expected)
        self.assertGreater(total, trainable)

    def test_forward_works_after_freeze(self):
        x = torch.randn(4, 27)
        for scope in SUPPORTED_TRAINABLE_SCOPES:
            model = PolicyValueNet(
                hidden_sizes=(96, 3),
                model_type="residual_v3",
                input_size=27,
            )
            apply_trainable_scope(model, scope)
            logits, value = model(x)
            self.assertEqual(logits.shape, (4, 6))
            self.assertEqual(value.shape, (4, 1))

    def test_reapply_all_after_other_scope_restores(self):
        apply_trainable_scope(self.model, "policy_head")
        total_pf, trainable_pf = _count_parameters(self.model)
        self.assertLess(trainable_pf, total_pf)

        apply_trainable_scope(self.model, "all")
        total, trainable = _count_parameters(self.model)
        self.assertEqual(total, trainable)
        self.assertEqual(total, total_pf)

    def test_policy_head_infers_gradients(self):
        apply_trainable_scope(self.model, "policy_head")
        optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=0.001,
        )
        x = torch.randn(4, 27)
        p_target = torch.ones(4, 6) / 6.0
        v_target = torch.zeros(4, 1)
        logits, value = self.model(x)
        policy_loss = -(p_target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = torch.square(value - v_target).mean()
        loss = policy_loss + 0.5 * value_loss
        loss.backward()
        optimizer.step()

    def test_last_block_policy_infers_gradients(self):
        apply_trainable_scope(self.model, "last_block_policy")
        optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=0.001,
        )
        x = torch.randn(4, 27)
        p_target = torch.ones(4, 6) / 6.0
        v_target = torch.zeros(4, 1)
        logits, value = self.model(x)
        policy_loss = -(p_target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = torch.square(value - v_target).mean()
        loss = policy_loss + 0.5 * value_loss
        loss.backward()
        optimizer.step()

    def test_scope_rejects_unsupported_value(self):
        with self.assertRaises(ValueError):
            apply_trainable_scope(self.model, "nonexistent")

    def test_scope_rejects_mlp_model(self):
        mlp = PolicyValueNet(
            hidden_sizes=(64, 64),
            model_type="mlp_v1",
            input_size=15,
        )
        with self.assertRaises(ValueError):
            apply_trainable_scope(mlp, "policy_head")
        with self.assertRaises(ValueError):
            apply_trainable_scope(mlp, "last_block_policy")


if __name__ == "__main__":
    unittest.main()
