from __future__ import annotations

from typing import Any

import torch

from ml.alphazero_lite import run_p1_policy_sensitivity_mapping as runner


def _ci(lower: float, upper: float) -> dict[str, Any]:
    return {
        "lower_95": lower,
        "upper_95": upper,
        "samples": 10_000,
        "unique_openings": 128,
    }


class TestSensitivityMapping:
    def test_interpolated_checkpoint_invariants(self) -> None:
        """Verify that alpha interpolation maintains trunk and value stack equality."""
        s0 = {
            "policy_hidden_layer.weight": torch.randn(96, 96, dtype=torch.float32),
            "policy_hidden_layer.bias": torch.randn(96, dtype=torch.float32),
            "policy_head.weight": torch.randn(6, 96, dtype=torch.float32),
            "policy_head.bias": torch.randn(6, dtype=torch.float32),
            "input_layer.weight": torch.randn(96, 27, dtype=torch.float32),
            "value_head.weight": torch.randn(1, 48, dtype=torch.float32),
        }
        delta = {
            "policy_hidden_layer.weight": torch.randn(96, 96, dtype=torch.float64)
            * 0.01,
            "policy_hidden_layer.bias": torch.randn(96, dtype=torch.float64) * 0.01,
            "policy_head.weight": torch.randn(6, 96, dtype=torch.float64) * 0.01,
            "policy_head.bias": torch.randn(6, dtype=torch.float64) * 0.01,
        }

        # Interpolate at alpha=0.5
        alpha = 0.5
        interpolated = {k: v.clone() for k, v in s0.items()}
        for k in runner.POLICY_KEYS:
            interpolated[k] = (s0[k].double() + alpha * delta[k]).float()

        # Check trunk and value equality
        assert runner.trunk_parameters_identical(interpolated, s0)
        assert runner.group_parameters_identical(
            interpolated, s0, runner.VALUE_STACK_PREFIXES
        )

    def test_ray_configs_structure(self) -> None:
        assert "ray1_p0_to_p1_to_y" in runner.RAY_CONFIGS
        assert "ray2_p1_to_p2" in runner.RAY_CONFIGS
        assert "ray3_p1_to_y" in runner.RAY_CONFIGS
        assert "ray4_p0_to_x" in runner.RAY_CONFIGS

        for ray_name, cfg in runner.RAY_CONFIGS.items():
            assert "base" in cfg
            assert "opponent" in cfg
            assert "delta" in cfg
            assert "alphas" in cfg
            assert len(cfg["alphas"]) > 0
