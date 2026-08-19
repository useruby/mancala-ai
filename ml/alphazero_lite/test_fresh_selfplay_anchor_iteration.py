from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ml.alphazero_lite import run_fresh_selfplay_anchor_iteration as runner


def _ci(lower: float, upper: float) -> dict[str, Any]:
    return {"lower_95": lower, "upper_95": upper, "samples": 10_000}


def _effect(effect: float, lower: float, upper: float) -> dict[str, Any]:
    return {
        "paired_candidate_effect": effect,
        "opening_bootstrap_ci": _ci(lower, upper),
        "p0_effect": effect,
        "p1_effect": 0.0,
        "win_draw_loss": {"wins": 20, "draws": 10, "losses": 10},
        "orientation": "candidate_minus_incumbent",
    }


def _probe_entry(fit: float | None, search_ce: float = 1.0) -> dict[str, Any]:
    return {
        "ce_candidate_search": search_ce,
        "ce_candidate_incumbent": 1.1,
        "ce_candidate_mixed": 1.05,
        "ce_incumbent_search": 1.2,
        "search_target_ce_improvement_vs_incumbent": 1.2 - search_ce,
        "fit_fraction": fit,
        "fit_denominator": 0.2,
        "drift_vs_incumbent": {
            "legal_l1_mean": 0.05,
            "legal_l1_max": 0.2,
            "legal_l1_p50": 0.04,
            "legal_l1_p90": 0.1,
            "legal_l1_p95": 0.12,
            "legal_l1_p99": 0.15,
            "legal_js_mean": 0.01,
            "top1_change_rate": 0.02,
        },
        "drift_vs_search_target": {
            "legal_l1_mean": 0.5,
            "legal_l1_max": 1.0,
            "legal_js_mean": 0.1,
            "top1_change_rate": 0.3,
        },
    }


def _mock_summary(
    *,
    lanes_start: bool = True,
    trunk_zero: bool = True,
    value_zero: bool = True,
    beta100_equiv: bool = True,
    arena: dict[str, Any] | None = None,
    fits: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    lanes = [name for name, _ in runner.BETA_LANES]
    fits = fits or {lane: 0.5 for lane in lanes}
    if arena is None:
        arena = {
            "beta_000": {"46": {"384:256": _effect(-0.20, -0.25, -0.15)}},
            "beta_095": {
                "46": {
                    "384:256": _effect(-0.01, -0.025, +0.005),
                    "1200:1200": _effect(+0.002, 0.000, +0.008),
                }
            },
            "beta_100": {"46": {"384:256": _effect(0.0, -0.005, +0.005)}},
        }
    return {
        "inputs": {
            "workdir": "/tmp/test",
            "current_weights_sha256": "current_sha",
            "fresh_selfplay": {
                "games_requested": 700,
                "replay_sha256": "fresh_sha",
            },
            "seed": 42,
            "optimizer": {"type": "Adam", "lr": 1e-5, "weight_decay": 0.0},
            "gradient_clip": 1.0,
        },
        "sanity": {
            "lanes_start_identical": lanes_start,
            "all_lanes_trunk_zero_change": trunk_zero,
            "all_lanes_value_stack_zero_change": value_zero,
            "beta_100_incumbent_equivalent": beta100_equiv,
            "beta_100_policy_head_drift": 0.00004,
            "beta_100_drift_tolerance": 0.001,
            "beta_100_initial_policy_grad_norm": 1e-6,
        },
        "dataset_shift_diagnostics": {
            "fresh": {
                "game_count": 700,
                "position_count": 28000,
                "game_length": {"mean": 40.0, "p50": 39.0, "p90": 55.0},
                "player_distribution": {"p0_fraction": 0.5},
                "search_policy_entropy": {"mean": 0.45},
                "incumbent_policy_entropy": {"mean": 0.95},
                "legal_l1_search_vs_incumbent": {
                    "mean": 0.8,
                    "p50": 0.7,
                    "p90": 1.6,
                    "p95": 1.7,
                    "p99": 1.9,
                },
                "legal_js_search_vs_incumbent": {
                    "mean": 0.17,
                    "p50": 0.12,
                    "p90": 0.42,
                    "p95": 0.49,
                    "p99": 0.58,
                },
                "top1_disagreement_rate": 0.40,
            },
            "historical": {
                "game_count": 700,
                "position_count": 27538,
                "game_length": {"mean": 39.3, "p50": 38.0, "p90": 54.0},
                "player_distribution": {"p0_fraction": 0.5},
                "search_policy_entropy": {"mean": 0.42},
                "incumbent_policy_entropy": {"mean": 0.94},
                "legal_l1_search_vs_incumbent": {
                    "mean": 0.81,
                    "p50": 0.74,
                    "p90": 1.61,
                    "p95": 1.75,
                    "p99": 1.90,
                },
                "legal_js_search_vs_incumbent": {
                    "mean": 0.174,
                    "p50": 0.123,
                    "p90": 0.425,
                    "p95": 0.492,
                    "p99": 0.588,
                },
                "top1_disagreement_rate": 0.405,
            },
        },
        "checkpoint_steps": [1, 4, 16, 46],
        "arena": arena,
        "probe_target_metrics": {
            lane: {"46": _probe_entry(fits.get(lane))} for lane in lanes
        },
        "drift": {
            lane: {
                "46": {
                    "trunk": 0.0,
                    "policy_head": 0.02,
                    "value_head": 0.0,
                }
            }
            for lane in lanes
        },
    }


def test_mixed_policy_target_contract() -> None:
    p_search = torch.tensor([[0.4, 0.6, 0.0, 0.0, 0.0, 0.0]])
    p_inc = torch.tensor([[0.1, 0.9, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 0.0]])

    # beta=0 is pure search
    assert torch.equal(runner.mixed_policy_target(p_search, p_inc, mask, 0.0), p_search)

    # beta=0.95 mixture
    out_095 = runner.mixed_policy_target(p_search, p_inc, mask, 0.95)
    expected_0 = 0.05 * 0.4 + 0.95 * 0.1  # 0.02 + 0.095 = 0.115
    expected_1 = 0.05 * 0.6 + 0.95 * 0.9  # 0.03 + 0.855 = 0.885
    assert float(out_095[0, 0]) == pytest.approx(expected_0)
    assert float(out_095[0, 1]) == pytest.approx(expected_1)
    assert float(out_095.sum()) == pytest.approx(1.0)

    # beta=1.0 is incumbent
    out_100 = runner.mixed_policy_target(p_search, p_inc, mask, 1.0)
    assert float(out_100[0, 0]) == pytest.approx(0.1)
    assert float(out_100[0, 1]) == pytest.approx(0.9)


def test_classify_fresh_safe_window_replicated() -> None:
    summary = _mock_summary(fits={"beta_000": 1.0, "beta_095": 0.45, "beta_100": 0.0})
    res = runner.classify_fresh(summary)
    assert res["label"] == "fresh_safe_window_replicated"
    assert "second AlphaZero-style iteration" in res["next_experiment"]


def test_classify_fresh_unanchored_update_safe() -> None:
    arena = {
        "beta_000": {"46": {"384:256": _effect(-0.005, -0.015, +0.005)}},
        "beta_095": {
            "46": {
                "384:256": _effect(-0.005, -0.015, +0.005),
                "1200:1200": _effect(+0.002, 0.0, +0.005),
            }
        },
        "beta_100": {"46": {"384:256": _effect(0.0, -0.005, +0.005)}},
    }
    summary = _mock_summary(
        arena=arena, fits={"beta_000": 1.0, "beta_095": 0.45, "beta_100": 0.0}
    )
    res = runner.classify_fresh(summary)
    assert res["label"] == "fresh_unanchored_update_safe"


def test_classify_fixed_replay_window_did_not_generalize() -> None:
    arena = {
        "beta_000": {"46": {"384:256": _effect(-0.20, -0.25, -0.15)}},
        "beta_095": {"46": {"384:256": _effect(-0.15, -0.20, -0.10)}},
        "beta_100": {"46": {"384:256": _effect(0.0, -0.005, +0.005)}},
    }
    summary = _mock_summary(
        arena=arena, fits={"beta_000": 1.0, "beta_095": 0.45, "beta_100": 0.0}
    )
    res = runner.classify_fresh(summary)
    assert res["label"] == "fixed_replay_window_did_not_generalize"


def test_classify_anchor_suppresses_fresh_learning() -> None:
    arena = {
        "beta_000": {"46": {"384:256": _effect(-0.20, -0.25, -0.15)}},
        "beta_095": {
            "46": {
                "384:256": _effect(-0.01, -0.025, +0.005),
                "1200:1200": _effect(+0.002, 0.000, +0.008),
            }
        },
        "beta_100": {"46": {"384:256": _effect(0.0, -0.005, +0.005)}},
    }
    summary = _mock_summary(
        arena=arena, fits={"beta_000": 1.0, "beta_095": 0.10, "beta_100": 0.0}
    )
    res = runner.classify_fresh(summary)
    assert res["label"] == "anchor_suppresses_fresh_learning"


def test_classify_invariant_failure_on_drift() -> None:
    summary = _mock_summary(trunk_zero=False)
    res = runner.classify_fresh(summary)
    assert res["label"] == "invariant_failure"

    summary_b100 = _mock_summary(beta100_equiv=False)
    res_b100 = runner.classify_fresh(summary_b100)
    assert res_b100["label"] == "invariant_failure"


def test_dataset_diagnostics_calculation() -> None:
    # Build 2 synthetic games with valid states
    rows = [
        {
            "game_index": 0,
            "player": 0,
            "move_index": 0,
            "value": 1.0,
            "state": [0.0833] * 12
            + [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.166,
                0.5,
                1.0,
                0.0,
                0.0,
                0.0,
                0.166,
                0.5,
            ],
            "policy": [0.0, 0.3, 0.3, 0.2, 0.1, 0.1],
        },
        {
            "game_index": 0,
            "player": 1,
            "move_index": 1,
            "value": -1.0,
            "state": [0.0833] * 12
            + [
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.166,
                0.5,
                1.0,
                0.0,
                0.0,
                0.0,
                0.166,
                0.5,
            ],
            "policy": [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
        },
        {
            "game_index": 1,
            "player": 0,
            "move_index": 0,
            "value": 0.0,
            "state": [0.0833] * 12
            + [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.166,
                0.5,
                1.0,
                0.0,
                0.0,
                0.0,
                0.166,
                0.5,
            ],
            "policy": [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
        },
    ]
    model = runner._new_model(torch.device("cpu"))
    incumbent_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    diag = runner.compute_dataset_diagnostics(rows, incumbent_state)

    assert diag["game_count"] == 2
    assert diag["position_count"] == 3
    assert "game_length" in diag
    assert "player_distribution" in diag
    assert "search_policy_entropy" in diag
    assert "incumbent_policy_entropy" in diag
    assert "legal_l1_search_vs_incumbent" in diag
    assert "legal_js_search_vs_incumbent" in diag


def test_markdown_report_renders() -> None:
    summary = _mock_summary(fits={"beta_000": 1.0, "beta_095": 0.45, "beta_100": 0.0})
    summary["classification"] = runner.classify_fresh(summary)
    md = runner.markdown_report(summary)
    assert "# AlphaZero-Lite Fresh Self-Play Anchor Validation Results" in md
    assert "Classification:" in md
    assert "Dataset-Shift Diagnostics" in md
    assert "Canonical Paired Arena" in md


def test_pr202_fixed_replay_smoke_check() -> None:
    """Smoke check on fixed replay target construction."""
    hist_manifest_path = Path(
        "/tmp/azlite_shared_trunk_learning/training_manifest.json"
    )
    if not hist_manifest_path.is_file():
        pytest.skip("Historical PR #191/#202 manifest not present")

    from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
        read_jsonl,
        verify_manifest,
    )
    from ml.alphazero_lite.train import legal_mask_matrix_for_encoded_states

    manifest = verify_manifest(hist_manifest_path)
    rows = read_jsonl(Path(manifest["replay_path"]))[:10]
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    p_search = torch.tensor([row["policy"] for row in rows], dtype=torch.float32)

    model = runner._new_model(torch.device("cpu"))
    incumbent_policy = runner.incumbent_policy_batch(
        model, {"x": torch.tensor(x), "mask": torch.tensor(mask)}
    )

    mixed_0 = runner.mixed_policy_target(
        p_search, incumbent_policy, torch.tensor(mask), 0.0
    )
    mixed_95 = runner.mixed_policy_target(
        p_search, incumbent_policy, torch.tensor(mask), 0.95
    )
    mixed_100 = runner.mixed_policy_target(
        p_search, incumbent_policy, torch.tensor(mask), 1.0
    )

    assert torch.equal(mixed_0, p_search)
    runner.assert_legal_distribution(mixed_95.numpy(), mask)
    runner.assert_legal_distribution(mixed_100.numpy(), mask)
