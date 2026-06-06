import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from ml.alphazero_lite.train import (
    POLICY_SIZE,
    SUPPORTED_TRAINABLE_SCOPES,
    PolicyValueNet,
    _count_parameters,
    apply_trainable_scope,
    checkpoint_from_model,
    load_checkpoint_into_model,
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


class ResidualV4MoveFactorizedTest(unittest.TestCase):
    def setUp(self):
        self.model = PolicyValueNet(
            hidden_sizes=(96, 3),
            model_type="residual_v4_move_factorized",
            input_size=27,
        )

    def test_forward_pass_shapes(self):
        x = torch.randn(4, 27)
        logits, value = self.model(x)
        self.assertEqual(logits.shape, (4, 6))
        self.assertEqual(value.shape, (4, 1))

    def test_move_projections_exist(self):
        self.assertIsNotNone(self.model.move_projections)
        self.assertEqual(len(self.model.move_projections), 6)
        for proj in self.model.move_projections:
            self.assertEqual(proj.weight.shape, (1, 96))
            self.assertEqual(proj.bias.shape, (1,))

    def test_no_global_policy_head(self):
        self.assertFalse(hasattr(self.model, "policy_head"))

    def test_policy_head_scope_freezes_trunk_and_value(self):
        apply_trainable_scope(self.model, "policy_head")
        trainable = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }
        self.assertIn("policy_hidden_layer.weight", trainable)
        self.assertIn("policy_hidden_layer.bias", trainable)
        for i in range(6):
            self.assertIn(f"move_projections.{i}.weight", trainable)
            self.assertIn(f"move_projections.{i}.bias", trainable)

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

    def test_last_block_policy_scope(self):
        apply_trainable_scope(self.model, "last_block_policy")
        trainable = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }
        self.assertIn("policy_hidden_layer.weight", trainable)
        for i in range(6):
            self.assertIn(f"move_projections.{i}.weight", trainable)

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
        self.assertIn("value_hidden_layer.weight", frozen)
        self.assertIn("value_hidden_layer.bias", frozen)
        self.assertIn("value_head.weight", frozen)
        self.assertIn("value_head.bias", frozen)

    def test_policy_head_scope_parameter_count(self):
        apply_trainable_scope(self.model, "policy_head")
        total, trainable = _count_parameters(self.model)
        expected = (
            96 * 96
            + 96  # policy_hidden: 9312
            + 6 * (96 * 1)
            + 6 * 1  # 6 move projections: 576 + 6
        )  # 9894
        self.assertEqual(trainable, expected)
        self.assertGreater(total, trainable)

    def test_last_block_policy_scope_parameter_count(self):
        apply_trainable_scope(self.model, "last_block_policy")
        total, trainable = _count_parameters(self.model)
        expected = (
            96 * 96
            + 96  # first layer of last block: 9312
            + 96 * 96
            + 96  # second layer of last block: 9312
            + 96 * 96
            + 96  # policy_hidden: 9312
            + 6 * 96
            + 6  # move projections: 582
        )  # 28518
        self.assertEqual(trainable, expected)
        self.assertGreater(total, trainable)

    def test_forward_works_after_freeze(self):
        x = torch.randn(4, 27)
        for scope in SUPPORTED_TRAINABLE_SCOPES:
            model = PolicyValueNet(
                hidden_sizes=(96, 3),
                model_type="residual_v4_move_factorized",
                input_size=27,
            )
            apply_trainable_scope(model, scope)
            logits, value = model(x)
            self.assertEqual(logits.shape, (4, 6))
            self.assertEqual(value.shape, (4, 1))

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

    def test_invalid_model_type_fails(self):
        with self.assertRaises(ValueError):
            PolicyValueNet(
                hidden_sizes=(96, 3),
                model_type="nonexistent",
                input_size=27,
            )


class ResidualV4CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.v3_model = PolicyValueNet(
            hidden_sizes=(96, 3),
            model_type="residual_v3",
            input_size=27,
        )
        self.v4_model = PolicyValueNet(
            hidden_sizes=(96, 3),
            model_type="residual_v4_move_factorized",
            input_size=27,
        )

    def test_v3_checkpoint_round_trip(self):
        ckpt = checkpoint_from_model(self.v3_model)
        self.assertIn("w_policy", ckpt)
        self.assertIn("b_policy", ckpt)
        self.assertIn("w_policy_hidden", ckpt)
        self.assertIn("w_value_hidden", ckpt)
        self.assertNotIn("w_policy_move_0", ckpt)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            np.savez(tmp_path, **ckpt)
            v3_loaded = PolicyValueNet(
                hidden_sizes=(96, 3),
                model_type="residual_v3",
                input_size=27,
            )
            skipped = load_checkpoint_into_model(v3_loaded, tmp_path)
            self.assertEqual(skipped, [])

            x = torch.randn(2, 27)
            with torch.no_grad():
                orig_logits, orig_value = self.v3_model(x)
                loaded_logits, loaded_value = v3_loaded(x)
            self.assertTrue(torch.allclose(orig_logits, loaded_logits, atol=1e-5))
            self.assertTrue(torch.allclose(orig_value, loaded_value, atol=1e-5))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_v4_checkpoint_round_trip(self):
        ckpt = checkpoint_from_model(self.v4_model)
        self.assertNotIn("w_policy", ckpt)
        self.assertNotIn("b_policy", ckpt)
        for i in range(POLICY_SIZE):
            self.assertIn(f"w_policy_move_{i}", ckpt)
            self.assertIn(f"b_policy_move_{i}", ckpt)
        self.assertIn("w_policy_hidden", ckpt)
        self.assertIn("w_value_hidden", ckpt)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            np.savez(tmp_path, **ckpt)
            v4_loaded = PolicyValueNet(
                hidden_sizes=(96, 3),
                model_type="residual_v4_move_factorized",
                input_size=27,
            )
            skipped = load_checkpoint_into_model(v4_loaded, tmp_path)
            self.assertEqual(skipped, [])

            x = torch.randn(2, 27)
            with torch.no_grad():
                orig_logits, orig_value = self.v4_model(x)
                loaded_logits, loaded_value = v4_loaded(x)
            self.assertTrue(torch.allclose(orig_logits, loaded_logits, atol=1e-5))
            self.assertTrue(torch.allclose(orig_value, loaded_value, atol=1e-5))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_v3_to_v4_partial_load_skips_policy_keys(self):
        ckpt = checkpoint_from_model(self.v3_model)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            np.savez(tmp_path, **ckpt)
            v4_target = PolicyValueNet(
                hidden_sizes=(96, 3),
                model_type="residual_v4_move_factorized",
                input_size=27,
            )
            original_move_weights = {
                i: v4_target.move_projections[i].weight.detach().clone()
                for i in range(POLICY_SIZE)
            }

            skipped = load_checkpoint_into_model(
                v4_target, tmp_path, report_skipped=True
            )
            self.assertIn("w_policy", skipped)
            self.assertIn("b_policy", skipped)

            for i in range(POLICY_SIZE):
                self.assertTrue(
                    torch.equal(
                        original_move_weights[i],
                        v4_target.move_projections[i].weight,
                    ),
                    f"move_projection {i} should retain random init after partial load",
                )

            self.assertTrue(
                torch.equal(
                    self.v3_model.input_layer.weight,
                    v4_target.input_layer.weight,
                ),
                "input_layer should be loaded from v3",
            )
            self.assertTrue(
                torch.equal(
                    self.v3_model.value_hidden_layer.weight,
                    v4_target.value_hidden_layer.weight,
                ),
                "value_hidden should be loaded from v3",
            )
            self.assertTrue(
                torch.equal(
                    self.v3_model.policy_hidden_layer.weight,
                    v4_target.policy_hidden_layer.weight,
                ),
                "policy_hidden should be loaded from v3",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_v4_checkpoint_loaded_into_v3_skips_move_keys(self):
        ckpt = checkpoint_from_model(self.v4_model)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            np.savez(tmp_path, **ckpt)
            v3_target = PolicyValueNet(
                hidden_sizes=(96, 3),
                model_type="residual_v3",
                input_size=27,
            )
            original_policy_weight = v3_target.policy_head.weight.detach().clone()

            skipped = load_checkpoint_into_model(
                v3_target, tmp_path, report_skipped=True
            )
            self.assertIn("w_policy_move_0", skipped)

            self.assertTrue(
                torch.equal(original_policy_weight, v3_target.policy_head.weight),
                "policy_head should retain random init",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_v3_export_artifact_behavior_unchanged(self):
        ckpt = checkpoint_from_model(self.v3_model)
        self.assertIn("w_policy", ckpt)
        self.assertIn("b_policy", ckpt)
        self.assertIn("w_policy_hidden", ckpt)
        self.assertIn("b_policy_hidden", ckpt)
        self.assertIn("w_value_hidden", ckpt)
        self.assertIn("b_value_hidden", ckpt)
        self.assertIn("w_input", ckpt)
        self.assertIn("b_input", ckpt)
        self.assertNotIn("w_policy_move_0", ckpt)

        for key in ckpt:
            self.assertIsInstance(ckpt[key], np.ndarray)
            self.assertEqual(ckpt[key].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
