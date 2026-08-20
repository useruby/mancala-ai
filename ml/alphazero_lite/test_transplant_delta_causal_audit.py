from __future__ import annotations

from typing import Any

import torch

from ml.alphazero_lite import run_transplant_delta_causal_audit as runner


def _ci(lower: float, upper: float) -> dict[str, Any]:
    return {
        "lower_95": lower,
        "upper_95": upper,
        "samples": 10_000,
        "unique_openings": 128,
    }


def _effect(
    paired: float,
    lower: float,
    upper: float,
    p0: float,
    p1: float = 0.0,
    orientation: str = "candidate_minus_opponent",
) -> dict[str, Any]:
    return {
        "paired_candidate_effect": paired,
        "opening_bootstrap_ci": _ci(lower, upper),
        "p0_effect": p0,
        "p1_effect": p1,
        "win_draw_loss": {"wins": 380, "draws": 40, "losses": 92},
        "orientation": orientation,
    }


def _mock_arena(
    *,
    p1_p0: tuple[float, float, float, float] = (-0.0137, -0.0293, 0.0039, -0.0156),
    p2_p1: tuple[float, float, float, float] = (-0.0957, -0.1270, -0.0645, -0.1914),
    x_p0: tuple[float, float, float, float] = (-0.0137, -0.0293, 0.0039, -0.0156),
    y_p1: tuple[float, float, float, float] = (-0.0273, -0.0469, -0.0117, -0.0547),
) -> dict[str, Any]:
    return {
        "p1_vs_p0": _effect(
            p1_p0[0], p1_p0[1], p1_p0[2], p1_p0[3], -0.0117, "p1_minus_p0"
        ),
        "p2_vs_p1": _effect(p2_p1[0], p2_p1[1], p2_p1[2], p2_p1[3], 0.0, "p2_minus_p1"),
        "x_vs_p0": _effect(x_p0[0], x_p0[1], x_p0[2], x_p0[3], -0.0117, "x_minus_p0"),
        "y_vs_p1": _effect(y_p1[0], y_p1[1], y_p1[2], y_p1[3], 0.0, "y_minus_p1"),
    }


def _mock_delta_stats(
    cos_sim: float = 0.960198,
    proj: float = 0.897564,
    res_frac: float = 0.279320,
) -> dict[str, Any]:
    return {
        "delta_01_l2_norm": 0.018733,
        "delta_12_l2_norm": 0.017511,
        "cosine_similarity": cos_sim,
        "projection_12_on_01": proj,
        "residual_norm": 0.004891,
        "residual_fraction": res_frac,
        "per_layer": {},
    }


class TestDeltaMath:
    def test_orthogonal_decomposition(self) -> None:
        """Test projection and residual orthogonality on synthetic vectors."""
        torch.manual_seed(42)
        v1 = torch.randn(100, dtype=torch.float64)
        v2 = torch.randn(100, dtype=torch.float64)

        norm1 = float(torch.norm(v1).item())

        proj = float((torch.dot(v2, v1) / (norm1**2)).item())
        res = v2 - proj * v1
        dot_res_v1 = float(torch.dot(res, v1).item())

        assert abs(dot_res_v1) < 1e-10
        recon = proj * v1 + res
        assert torch.allclose(recon, v2, atol=1e-12)

    def test_compute_distribution_metrics(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = runner.compute_distribution_metrics(values)
        assert stats["mean"] == 3.0
        assert stats["max"] == 5.0
        assert stats["p50"] == 3.0
        assert stats["p90"] == 4.6
        assert stats["p99"] == 4.96

    def test_compute_distribution_metrics_empty(self) -> None:
        stats = runner.compute_distribution_metrics([])
        assert stats["mean"] == 0.0
        assert stats["max"] == 0.0


class TestTransplantInvariants:
    def test_delta_transplant_reconstruction(self) -> None:
        """Verify that P0 + delta_01 produces P1 bit-for-bit in float64."""
        s0 = {
            "policy_hidden_layer.weight": torch.randn(96, 96, dtype=torch.float32),
            "policy_hidden_layer.bias": torch.randn(96, dtype=torch.float32),
            "policy_head.weight": torch.randn(6, 96, dtype=torch.float32),
            "policy_head.bias": torch.randn(6, dtype=torch.float32),
            "input_layer.weight": torch.randn(96, 27, dtype=torch.float32),
            "value_head.weight": torch.randn(1, 48, dtype=torch.float32),
        }
        # Perturb policy head only
        s1 = {k: v.clone() for k, v in s0.items()}
        for k in runner.POLICY_KEYS:
            s1[k] = s0[k] + torch.randn_like(s0[k]) * 0.01

        s2 = {k: v.clone() for k, v in s1.items()}
        for k in runner.POLICY_KEYS:
            s2[k] = s1[k] + torch.randn_like(s1[k]) * 0.01

        d01, d12, stats = runner.compute_policy_deltas(s0, s1, s2)
        assert stats["cosine_similarity"] is not None

        # Control P1 reconstruction
        control_p1 = {k: v.clone() for k, v in s0.items()}
        for k in runner.POLICY_KEYS:
            control_p1[k] = (s0[k].double() + d01[k]).float()

        for k in s1:
            assert torch.equal(control_p1[k], s1[k])

        # Control P2 reconstruction
        control_p2 = {k: v.clone() for k, v in s1.items()}
        for k in runner.POLICY_KEYS:
            control_p2[k] = (s1[k].double() + d12[k]).float()

        for k in s2:
            assert torch.equal(control_p2[k], s2[k])


class TestClassification:
    def test_classify_p1_local_fragility(self) -> None:
        """X is safe on P0, Y regresses on P1 with P0-seat failure signature."""
        arena = _mock_arena(
            x_p0=(-0.0137, -0.0293, +0.0039, -0.0156),
            y_p1=(-0.0273, -0.0469, -0.0117, -0.0547),
            p2_p1=(-0.0957, -0.1270, -0.0645, -0.1914),
        )
        delta_stats = _mock_delta_stats()
        result = runner.classify_transplant_results(arena, delta_stats, {})
        assert result["label"] == "p1_local_policy_fragility"
        assert result["evidence"]["x_vs_p0_safe"] is True
        assert result["evidence"]["y_vs_p1_safe"] is False

    def test_classify_gen2_delta_intrinsically_toxic(self) -> None:
        """X materially regresses on P0, while Y is safe on P1."""
        arena = _mock_arena(
            x_p0=(-0.0800, -0.1100, -0.0500, -0.1500),
            y_p1=(-0.0050, -0.0200, +0.0100, -0.0100),
            p2_p1=(-0.0957, -0.1270, -0.0645, -0.1914),
        )
        delta_stats = _mock_delta_stats()
        result = runner.classify_transplant_results(arena, delta_stats, {})
        assert result["label"] == "gen2_delta_intrinsically_toxic"

    def test_classify_gen2_delta_parent_interaction(self) -> None:
        """Both X and Y are safe individually, but P2 (P1 + delta_12) regresses."""
        arena = _mock_arena(
            x_p0=(-0.0050, -0.0200, +0.0100, -0.0100),
            y_p1=(-0.0050, -0.0200, +0.0100, -0.0100),
            p2_p1=(-0.0957, -0.1270, -0.0645, -0.1914),
        )
        delta_stats = _mock_delta_stats()
        result = runner.classify_transplant_results(arena, delta_stats, {})
        assert result["label"] == "gen2_delta_parent_interaction"

    def test_classify_inconclusive(self) -> None:
        """All are safe or unclassifiable."""
        arena = _mock_arena(
            x_p0=(-0.0050, -0.0200, +0.0100, -0.0100),
            y_p1=(-0.0050, -0.0200, +0.0100, -0.0100),
            p2_p1=(-0.0050, -0.0200, +0.0100, -0.0100),
        )
        delta_stats = _mock_delta_stats()
        result = runner.classify_transplant_results(arena, delta_stats, {})
        assert result["label"] == "inconclusive"
