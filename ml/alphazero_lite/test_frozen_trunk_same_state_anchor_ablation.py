from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.alphazero_lite import run_frozen_trunk_same_state_anchor_ablation as runner


def _ci(lower: float, upper: float) -> dict:
    return {"lower_95": lower, "upper_95": upper, "samples": 10_000}


def _effect(effect: float, lower: float, upper: float) -> dict:
    return {
        "paired_candidate_effect": effect,
        "opening_bootstrap_ci": _ci(lower, upper),
        "p0_effect": effect,
        "p1_effect": 0.0,
        "win_draw_loss": {"wins": 0, "draws": 0, "losses": 0},
        "orientation": "candidate_minus_incumbent",
    }


def _probe_entry(fit: float | None, search_ce: float = 1.0) -> dict:
    return {
        "ce_candidate_search": search_ce,
        "ce_candidate_incumbent": 1.1,
        "ce_candidate_mixed": 1.05,
        "ce_incumbent_search": 1.2,
        "search_target_ce_improvement_vs_incumbent": 1.2 - search_ce,
        "fit_fraction": fit,
        "fit_denominator": 0.2,
        "drift_vs_incumbent": {},
        "drift_vs_search_target": {},
    }


def _full_summary(
    *,
    beta000_repro: bool = True,
    beta100_equiv: bool = True,
    arena: dict | None = None,
    fits: dict[str, float | None] | None = None,
) -> dict:
    lanes = [name for name, _ in runner.BETA_LANES]
    fits = fits or {lane: 0.5 for lane in lanes}
    if arena is None:
        arena = {
            lane: {"46": {"384:256": _effect(-0.1, -0.15, -0.05)}} for lane in lanes
        }
    return {
        "deterministic_reproduction": {"pr200_policy_head_state_hashes": beta000_repro},
        "sanity": {"beta_100_incumbent_equivalent": beta100_equiv},
        "arena": arena,
        "probe_target_metrics": {
            lane: {"46": _probe_entry(fits.get(lane))} for lane in lanes
        },
    }


def test_mixed_policy_target_beta_zero_is_byte_identical_to_search() -> None:
    p_search = torch.tensor([[0.4, 0.6, 0.0, 0.0, 0.0, 0.0]])
    p_inc = torch.tensor([[0.1, 0.9, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
    out = runner.mixed_policy_target(p_search, p_inc, mask, 0.0)
    assert torch.equal(out, p_search)


def test_mixed_policy_target_renormalizes_and_zeros_illegal() -> None:
    p_search = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])
    p_inc = torch.tensor([[0.3, 0.7, 0.0, 0.0, 0.0, 0.0]])
    # illegal moves 2,3,5; beta=0.5 -> (0.4, 0.6) over legal {0,1,4} but 4 is 0
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
    out = runner.mixed_policy_target(p_search, p_inc, mask, 0.5)
    assert float(out.sum()) == pytest.approx(1.0)
    assert float(out[0, 3]) == 0.0
    assert float(out[0, 5]) == 0.0
    assert float(out[0, 0]) == pytest.approx(0.4)
    assert float(out[0, 1]) == pytest.approx(0.6)


def test_mixed_policy_target_beta_one_is_incumbent() -> None:
    p_search = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])
    p_inc = torch.tensor([[0.2, 0.8, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
    out = runner.mixed_policy_target(p_search, p_inc, mask, 1.0)
    assert float(out[0, 0]) == pytest.approx(0.2)
    assert float(out[0, 1]) == pytest.approx(0.8)


def test_mixed_policy_target_defensive_fallback_on_all_zero_legal() -> None:
    p_search = torch.zeros((1, 6))
    p_inc = torch.zeros((1, 6))
    # four legal moves (indices 0, 1, 2, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
    out = runner.mixed_policy_target(p_search, p_inc, mask, 0.5)
    assert float(out.sum()) == pytest.approx(1.0)
    assert float(out[0, 3]) == 0.0
    assert float(out[0, 0]) == pytest.approx(0.25)


def test_assert_legal_distribution_rejects_illegal_mass() -> None:
    p = np.array([[0.5, 0.5, 0.0]])
    mask = np.array([[1.0, 1.0, 0.0]])
    # legal sum ok, but add illegal mass
    bad = p.copy()
    bad[0, 2] = 0.3
    import pytest

    with pytest.raises(AssertionError):
        runner.assert_legal_distribution(bad, mask)
    # valid distribution passes
    runner.assert_legal_distribution(p, mask)


def test_arena_safe_gate() -> None:
    # CI includes zero -> safe
    assert runner.arena_safe(_effect(0.01, -0.02, 0.04))
    # CI upper < 0 but lower >= -0.03 -> safe (noninferiority)
    assert runner.arena_safe(_effect(-0.02, -0.03, -0.01))
    # CI upper < 0 and lower < -0.03 -> not safe
    assert not runner.arena_safe(_effect(-0.20, -0.25, -0.15))


def test_classify_invariant_failure_on_missing_repro() -> None:
    summary = _full_summary(beta000_repro=False)
    result = runner.classify(summary)
    assert result["label"] == "invariant_failure"
    assert result["evidence"]["beta_000_reproduces_pr200"] is False


def test_classify_invariant_failure_on_beta100_drift() -> None:
    summary = _full_summary(beta100_equiv=False)
    result = runner.classify(summary)
    assert result["label"] == "invariant_failure"
    assert result["evidence"]["beta_100_incumbent_equivalent"] is False


def test_classify_safe_learning_window_found() -> None:
    lanes = [name for name, _ in runner.BETA_LANES]
    arena = {lane: {"46": {"384:256": _effect(-0.20, -0.25, -0.15)}} for lane in lanes}
    # beta_050 is safe and fits
    arena["beta_050"]["46"]["384:256"] = _effect(-0.01, -0.02, 0.01)
    fits = {
        "beta_000": 0.9,
        "beta_050": 0.5,
        "beta_080": 0.3,
        "beta_095": 0.1,
        "beta_100": None,
    }
    summary = _full_summary(arena=arena, fits=fits)
    result = runner.classify(summary)
    assert result["label"] == "safe_learning_window_found"
    assert "beta_050" in result["evidence"]["safe_window_lanes"]


def test_classify_same_state_anchor_insufficient() -> None:
    lanes = [name for name, _ in runner.BETA_LANES]
    # All beta<1 lanes materially negative but with meaningful fit
    arena = {lane: {"46": {"384:256": _effect(-0.20, -0.25, -0.15)}} for lane in lanes}
    fits = {
        "beta_000": 0.9,
        "beta_050": 0.6,
        "beta_080": 0.4,
        "beta_095": 0.3,
        "beta_100": None,
    }
    summary = _full_summary(arena=arena, fits=fits)
    result = runner.classify(summary)
    assert result["label"] == "same_state_anchor_insufficient"


def test_classify_anchor_only_freezes_learning() -> None:
    lanes = [name for name, _ in runner.BETA_LANES]
    arena = {lane: {"46": {"384:256": _effect(-0.20, -0.25, -0.15)}} for lane in lanes}
    # only beta_095 (heaviest anchor) is safe, but no anchored lane fits;
    # beta_100 is incumbent-equivalent (effect ~ 0).
    arena["beta_095"]["46"]["384:256"] = _effect(-0.01, -0.02, 0.01)
    arena["beta_100"]["46"]["384:256"] = _effect(0.0, 0.0, 0.0)
    fits = {
        "beta_000": 1.0,
        "beta_050": 0.1,
        "beta_080": 0.1,
        "beta_095": 0.1,
        "beta_100": None,
    }
    summary = _full_summary(arena=arena, fits=fits)
    result = runner.classify(summary)
    assert result["label"] == "anchor_only_freezes_learning"


def test_classify_non_monotonic_anchor_response() -> None:
    lanes = [name for name, _ in runner.BETA_LANES]
    # No anchored lane is safe and none fits; effects are non-monotonic.
    effects = {
        "beta_000": -0.20,
        "beta_050": -0.15,
        "beta_080": -0.19,  # non-monotonic dip
        "beta_095": -0.10,
        "beta_100": 0.0,
    }
    arena = {
        lane: {"46": {"384:256": _effect(e, e - 0.05, e + 0.05)}}
        for lane, e in effects.items()
    }
    fits = {lane: 0.1 for lane in lanes}
    fits["beta_000"] = 1.0
    fits["beta_100"] = None
    summary = _full_summary(arena=arena, fits=fits)
    result = runner.classify(summary)
    assert result["label"] == "non_monotonic_anchor_response"


def test_classify_safe_window_blocked_by_high_budget_regression() -> None:
    lanes = [name for name, _ in runner.BETA_LANES]
    arena = {lane: {"46": {"384:256": _effect(-0.20, -0.25, -0.15)}} for lane in lanes}
    arena["beta_050"]["46"]["384:256"] = _effect(-0.01, -0.02, 0.01)
    # high-budget regression for beta_050
    arena["beta_050"]["46"]["1200:1200"] = _effect(-0.10, -0.13, -0.07)
    fits = {
        "beta_000": 0.9,
        "beta_050": 0.5,
        "beta_080": 0.3,
        "beta_095": 0.1,
        "beta_100": None,
    }
    summary = _full_summary(arena=arena, fits=fits)
    result = runner.classify(summary)
    # beta_050 has meaningful fit but is not safe at 1200:1200; still safe at
    # 384:256 though, so the safe window is blocked -> not safe_learning_window.
    assert result["label"] != "safe_learning_window_found"


def test_policy_drift_metrics_percentiles_and_top1() -> None:
    cand = np.array([[0.5, 0.5, 0.0], [0.9, 0.1, 0.0]])
    ref = np.array([[0.5, 0.5, 0.0], [0.1, 0.9, 0.0]])
    mask = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    m = runner.policy_drift_metrics(cand, ref, mask)
    assert m["legal_l1_mean"] > 0.0
    assert m["legal_l1_max"] >= m["legal_l1_mean"]
    assert m["legal_l1_p50"] <= m["legal_l1_p90"]
    assert m["top1_change_rate"] == pytest.approx(0.5)


def test_incumbent_policy_batch_matches_legal_masking() -> None:
    from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _new_model
    from ml.alphazero_lite.train import load_checkpoint_into_model
    from pathlib import Path
    import numpy as np
    from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
        verify_manifest,
    )

    manifest = verify_manifest(
        Path("/tmp/azlite_shared_trunk_learning/training_manifest.json")
    )
    init = Path(manifest["artifact_paths"]["initialization_checkpoint"])
    model = _new_model(torch.device("cpu"))
    load_checkpoint_into_model(model, init)
    x = np.asarray([[1.0] * 27, [0.0] * 27], dtype=np.float32)
    mask = np.asarray(
        [[1.0, 1.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    batch = {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask)}
    p_inc = runner.incumbent_policy_batch(model, batch).numpy()
    assert np.allclose(p_inc.sum(axis=1), 1.0, atol=1e-5)
    assert float(p_inc[0, 3]) == 0.0
    assert float(p_inc[0, 5]) == 0.0
    assert float(p_inc[1, 1]) == 0.0
