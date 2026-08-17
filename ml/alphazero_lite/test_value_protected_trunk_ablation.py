"""Invariant tests for the one-sided shared-trunk gradient projection."""

from __future__ import annotations

import unittest

import torch

from ml.alphazero_lite.run_value_protected_trunk_ablation import (
    compose_gradients,
    is_trunk_parameter,
    parameter_family,
    project_policy_gradient,
)


class ValueProtectedTrunkAblationTest(unittest.TestCase):
    def test_positive_dot_leaves_policy_unchanged(self) -> None:
        policy = [torch.tensor([1.0, 2.0])]
        value = [torch.tensor([2.0, 1.0])]

        projected, telemetry = project_policy_gradient(policy, value, enabled=True)

        self.assertTrue(torch.equal(policy[0], projected[0]))
        self.assertFalse(telemetry["projection_fired"])

    def test_negative_dot_is_orthogonal_after_projection(self) -> None:
        policy = [torch.tensor([1.0, -2.0])]
        value = [torch.tensor([-3.0, 1.0])]

        projected, telemetry = project_policy_gradient(policy, value, enabled=True)

        self.assertTrue(telemetry["projection_fired"])
        self.assertAlmostEqual(0.0, float(torch.dot(projected[0], value[0])), places=6)

    def test_composition_preserves_private_and_value_gradients(self) -> None:
        parameters = [
            ("input_layer.weight", torch.nn.Parameter(torch.zeros(2))),
            ("residual_layers.0.0.bias", torch.nn.Parameter(torch.zeros(2))),
            ("policy_head.bias", torch.nn.Parameter(torch.zeros(2))),
            ("value_head.bias", torch.nn.Parameter(torch.zeros(1))),
        ]
        policy = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, -2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([0.0]),
        ]
        value = [
            torch.tensor([-1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([7.0]),
        ]
        value_before = [gradient.numpy().tobytes() for gradient in value]

        compose_gradients(parameters, policy, value, projection_enabled=True)

        self.assertEqual(
            value_before, [gradient.numpy().tobytes() for gradient in value]
        )
        self.assertTrue(torch.equal(policy[2], parameters[2][1].grad))
        self.assertTrue(torch.equal(value[3], parameters[3][1].grad))
        self.assertFalse(torch.equal(policy[0] + value[0], parameters[0][1].grad))
        self.assertTrue(
            all(is_trunk_parameter(name) for name, _parameter in parameters[:2])
        )

    def test_only_shared_trunk_can_be_projected(self) -> None:
        self.assertEqual("trunk", parameter_family("input_layer.bias"))
        self.assertEqual("trunk", parameter_family("residual_layers.2.1.weight"))
        self.assertEqual("policy_head", parameter_family("policy_hidden_layer.weight"))
        self.assertEqual("value_head", parameter_family("value_head.bias"))

    def test_disabled_composition_is_exact_unprojected_sum(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        named = [("input_layer.weight", parameter)]
        policy, value = [torch.tensor([3.0, -2.0])], [torch.tensor([-1.0, 1.0])]

        telemetry = compose_gradients(named, policy, value, projection_enabled=False)

        self.assertFalse(telemetry["projection_fired"])
        self.assertTrue(torch.equal(policy[0] + value[0], parameter.grad))

    def test_clipping_is_after_composition(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        named = [("input_layer.weight", parameter)]
        policy, value = [torch.tensor([10.0, 0.0])], [torch.tensor([-9.0, 1.0])]

        compose_gradients(named, policy, value, projection_enabled=True)
        composed = parameter.grad.clone()
        torch.nn.utils.clip_grad_norm_([parameter], 1.0)

        self.assertGreater(float(torch.linalg.vector_norm(composed)), 1.0)
        self.assertAlmostEqual(
            1.0, float(torch.linalg.vector_norm(parameter.grad)), places=6
        )


if __name__ == "__main__":
    unittest.main()
