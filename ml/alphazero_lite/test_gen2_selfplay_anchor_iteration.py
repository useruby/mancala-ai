from __future__ import annotations

from typing import Any

from ml.alphazero_lite import run_gen2_selfplay_anchor_iteration as runner


def _ci(lower: float, upper: float) -> dict[str, Any]:
    return {
        "lower_95": lower,
        "upper_95": upper,
        "samples": 10_000,
        "unique_openings": 128,
    }


def _effect(
    effect: float,
    lower: float,
    upper: float,
    orientation: str = "candidate_minus_parent",
) -> dict[str, Any]:
    return {
        "paired_candidate_effect": effect,
        "opening_bootstrap_ci": _ci(lower, upper),
        "p0_effect": effect,
        "p1_effect": 0.0,
        "win_draw_loss": {"wins": 200, "draws": 100, "losses": 100},
        "orientation": orientation,
    }


def _probe_entry(fit: float | None, search_ce: float = 1.0) -> dict[str, Any]:
    return {
        "ce_candidate_search": search_ce,
        "ce_candidate_p1": 1.1,
        "ce_candidate_p0": 1.15,
        "ce_candidate_mixed": 1.05,
        "ce_p1_search": 1.2,
        "ce_p0_search": 1.25,
        "search_target_ce_improvement_vs_p1": 1.2 - search_ce,
        "fit_fraction": fit,
        "fit_denominator": 0.2,
        "drift_vs_p1": {
            "legal_l1_mean": 0.05,
            "legal_l1_max": 0.2,
            "legal_l1_p50": 0.04,
            "legal_l1_p90": 0.1,
            "legal_l1_p95": 0.12,
            "legal_l1_p99": 0.15,
            "legal_js_mean": 0.01,
            "top1_change_rate": 0.02,
        },
        "drift_vs_p0": {
            "legal_l1_mean": 0.08,
            "legal_l1_max": 0.25,
            "legal_l1_p50": 0.06,
            "legal_l1_p90": 0.15,
            "legal_l1_p95": 0.18,
            "legal_l1_p99": 0.22,
            "legal_js_mean": 0.02,
            "top1_change_rate": 0.03,
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
    trunk_zero_p1: bool = True,
    value_zero_p1: bool = True,
    trunk_zero_p0: bool = True,
    value_zero_p0: bool = True,
    beta100_equiv: bool = True,
    p1_reproduced: bool = True,
    arena: dict[str, Any] | None = None,
    fits: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    lanes = [name for name, _ in runner.BETA_LANES]
    fits = fits or {lane: 0.5 for lane in lanes}
    if arena is None:
        arena = {
            "p1_vs_p0": {
                "384:256": _effect(-0.0137, -0.0293, +0.0039, "candidate_minus_p0"),
                "1200:1200": _effect(+0.0234, +0.0078, +0.0430, "candidate_minus_p0"),
            },
            "vs_p1": {
                "beta_000": {
                    "46": {
                        "384:256": _effect(-0.06, -0.09, -0.03, "candidate_minus_p1")
                    }
                },
                "beta_095": {
                    "16": {
                        "384:256": _effect(-0.01, -0.02, +0.00, "candidate_minus_p1")
                    },
                    "46": {
                        "384:256": _effect(-0.01, -0.025, +0.005, "candidate_minus_p1"),
                        "1200:1200": _effect(
                            +0.02, +0.005, +0.035, "candidate_minus_p1"
                        ),
                    },
                },
                "beta_100": {
                    "46": {
                        "384:256": _effect(0.0, -0.005, +0.005, "candidate_minus_p1")
                    }
                },
            },
            "vs_p0": {
                "beta_095": {
                    "46": {
                        "384:256": _effect(-0.01, -0.025, +0.005, "candidate_minus_p0"),
                        "1200:1200": _effect(
                            +0.03, +0.010, +0.050, "candidate_minus_p0"
                        ),
                    }
                }
            },
        }
    return {
        "inputs": {
            "workdir": "/tmp/test",
            "p0_weights_sha256": runner.P0_EXPECTED_HASH,
            "p1_weights_sha256": "p1_weights_sha",
            "p1_checkpoint_npz_sha256": runner.P1_EXPECTED_NPZ_HASH,
            "p1_state_hash": runner.P1_EXPECTED_STATE_HASH,
            "gen1_replay_sha256": runner.PR203_REPLAY_HASH,
            "gen2_selfplay": {
                "games_requested": 700,
                "replay_sha256": "gen2_replay_sha",
            },
            "seed": 43,
            "optimizer": {"type": "Adam", "lr": 1e-5, "weight_decay": 0.0},
            "gradient_clip": 1.0,
            "trainable_scope": "policy_head",
        },
        "sanity": {
            "p1_reconstructed_and_verified": p1_reproduced,
            "lanes_start_identical": lanes_start,
            "all_lanes_trunk_zero_change_vs_p1": trunk_zero_p1,
            "all_lanes_value_stack_zero_change_vs_p1": value_zero_p1,
            "all_lanes_trunk_zero_change_vs_p0": trunk_zero_p0,
            "all_lanes_value_stack_zero_change_vs_p0": value_zero_p0,
            "beta_100_p1_equivalent": beta100_equiv,
            "beta_100_p1_drift": 0.00004,
            "beta_100_drift_tolerance": 0.001,
            "beta_100_initial_policy_grad_norm": 1e-6,
        },
        "dataset_evolution_diagnostics": {
            "gen2_p1_selfplay": {
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
            "gen1_p0_selfplay": {
                "game_count": 700,
                "position_count": 28146,
                "game_length": {"mean": 40.2, "p50": 41.0, "p90": 52.0},
                "player_distribution": {"p0_fraction": 0.5176},
                "search_policy_entropy": {"mean": 0.33},
                "incumbent_policy_entropy": {"mean": 0.935},
                "legal_l1_search_vs_incumbent": {
                    "mean": 0.869,
                    "p50": 0.832,
                    "p90": 1.668,
                    "p95": 1.786,
                    "p99": 1.919,
                },
                "legal_js_search_vs_incumbent": {
                    "mean": 0.195,
                    "p50": 0.153,
                    "p90": 0.451,
                    "p95": 0.516,
                    "p99": 0.606,
                },
                "top1_disagreement_rate": 0.4087,
            },
            "deltas_gen2_minus_gen1": {
                "game_count_delta": 0,
                "position_count_delta": -146,
                "game_length_mean_delta": -0.2,
                "search_entropy_mean_delta": 0.12,
                "parent_entropy_mean_delta": 0.015,
                "l1_mean_delta": -0.069,
                "js_mean_delta": -0.025,
                "top1_disagreement_delta": -0.0087,
            },
        },
        "checkpoint_steps": [1, 4, 16, 46],
        "arena": arena,
        "probe_target_metrics": {
            lane: {"46": _probe_entry(fits.get(lane))} for lane in lanes
        },
        "drift_vs_p1": {
            lane: {
                "46": {
                    "trunk": 0.0,
                    "policy_head": 0.02,
                    "value_head": 0.0,
                }
            }
            for lane in lanes
        },
        "drift_vs_p0": {
            lane: {
                "46": {
                    "trunk": 0.0,
                    "policy_head": 0.03,
                    "value_head": 0.0,
                }
            }
            for lane in lanes
        },
    }


def test_classify_cumulative_safe_gain() -> None:
    summary = _mock_summary(fits={"beta_000": 1.0, "beta_095": 0.55, "beta_100": 0.0})
    res = runner.classify_gen2(summary)
    assert res["label"] == "cumulative_safe_gain"
    assert "short 3-5 generation lineage experiment" in res["next_experiment"]


def test_classify_safe_second_step_gain_unproven() -> None:
    # 1200:1200 CI touches/crosses zero, but is safe (lower >= -0.03)
    arena = {
        "p1_vs_p0": {"384:256": _effect(-0.0137, -0.0293, +0.0039)},
        "vs_p1": {
            "beta_000": {"46": {"384:256": _effect(-0.06, -0.09, -0.03)}},
            "beta_095": {
                "46": {
                    "384:256": _effect(-0.01, -0.025, +0.005),
                    "1200:1200": _effect(+0.005, -0.010, +0.020),
                }
            },
            "beta_100": {"46": {"384:256": _effect(0.0, -0.005, +0.005)}},
        },
        "vs_p0": {
            "beta_095": {
                "46": {
                    "384:256": _effect(-0.01, -0.025, +0.005),
                    "1200:1200": _effect(+0.01, -0.005, +0.025),
                }
            }
        },
    }
    summary = _mock_summary(
        arena=arena, fits={"beta_000": 1.0, "beta_095": 0.55, "beta_100": 0.0}
    )
    res = runner.classify_gen2(summary)
    assert res["label"] == "safe_second_step_gain_unproven"
    assert "increase evaluation power" in res["next_experiment"]


def test_classify_second_iteration_regression() -> None:
    arena = {
        "p1_vs_p0": {"384:256": _effect(-0.0137, -0.0293, +0.0039)},
        "vs_p1": {
            "beta_000": {"46": {"384:256": _effect(-0.06, -0.09, -0.03)}},
            "beta_095": {
                "46": {
                    "384:256": _effect(-0.05, -0.08, -0.02),
                }
            },
            "beta_100": {"46": {"384:256": _effect(0.0, -0.005, +0.005)}},
        },
        "vs_p0": {},
    }
    summary = _mock_summary(
        arena=arena, fits={"beta_000": 1.0, "beta_095": 0.55, "beta_100": 0.0}
    )
    res = runner.classify_gen2(summary)
    assert res["label"] == "second_iteration_regression"
    assert "disagreement tails" in res["next_experiment"]


def test_classify_anchor_learning_signal_collapsed() -> None:
    summary = _mock_summary(fits={"beta_000": 1.0, "beta_095": 0.15, "beta_100": 0.0})
    res = runner.classify_gen2(summary)
    assert res["label"] == "anchor_learning_signal_collapsed"
    assert "adaptive parent-relative step sizing" in res["next_experiment"]


def test_classify_unanchored_second_generation_safe() -> None:
    arena = {
        "p1_vs_p0": {"384:256": _effect(-0.0137, -0.0293, +0.0039)},
        "vs_p1": {
            "beta_000": {"46": {"384:256": _effect(-0.005, -0.015, +0.005)}},
            "beta_095": {
                "46": {
                    "384:256": _effect(-0.005, -0.015, +0.005),
                    "1200:1200": _effect(+0.02, +0.005, +0.035),
                }
            },
            "beta_100": {"46": {"384:256": _effect(0.0, -0.005, +0.005)}},
        },
        "vs_p0": {
            "beta_095": {
                "46": {
                    "384:256": _effect(-0.01, -0.025, +0.005),
                    "1200:1200": _effect(+0.03, +0.010, +0.050),
                }
            }
        },
    }
    summary = _mock_summary(
        arena=arena, fits={"beta_000": 1.0, "beta_095": 0.55, "beta_100": 0.0}
    )
    res = runner.classify_gen2(summary)
    assert res["label"] == "unanchored_second_generation_safe"


def test_classify_invariant_failures() -> None:
    assert (
        runner.classify_gen2(_mock_summary(lanes_start=False))["label"]
        == "invariant_failure"
    )
    assert (
        runner.classify_gen2(_mock_summary(trunk_zero_p1=False))["label"]
        == "invariant_failure"
    )
    assert (
        runner.classify_gen2(_mock_summary(value_zero_p1=False))["label"]
        == "invariant_failure"
    )
    assert (
        runner.classify_gen2(_mock_summary(trunk_zero_p0=False))["label"]
        == "invariant_failure"
    )
    assert (
        runner.classify_gen2(_mock_summary(value_zero_p0=False))["label"]
        == "invariant_failure"
    )
    assert (
        runner.classify_gen2(_mock_summary(beta100_equiv=False))["label"]
        == "invariant_failure"
    )
    assert (
        runner.classify_gen2(_mock_summary(p1_reproduced=False))["label"]
        == "invariant_failure"
    )


def test_dataset_evolution_comparison() -> None:
    gen2_diag = {
        "game_count": 700,
        "position_count": 28000,
        "game_length": {"mean": 40.0},
        "search_policy_entropy": {"mean": 0.45},
        "incumbent_policy_entropy": {"mean": 0.95},
        "legal_l1_search_vs_incumbent": {"mean": 0.8},
        "legal_js_search_vs_incumbent": {"mean": 0.17},
        "top1_disagreement_rate": 0.40,
    }
    gen1_diag = {
        "game_count": 700,
        "position_count": 28146,
        "game_length": {"mean": 40.2},
        "search_policy_entropy": {"mean": 0.33},
        "incumbent_policy_entropy": {"mean": 0.935},
        "legal_l1_search_vs_incumbent": {"mean": 0.869},
        "legal_js_search_vs_incumbent": {"mean": 0.195},
        "top1_disagreement_rate": 0.4087,
    }
    comp = runner.dataset_evolution_comparison(gen2_diag, gen1_diag)
    deltas = comp["deltas_gen2_minus_gen1"]
    assert deltas["game_count_delta"] == 0
    assert deltas["position_count_delta"] == -146
    assert abs(deltas["game_length_mean_delta"] - (-0.2)) < 1e-6
    assert abs(deltas["search_entropy_mean_delta"] - 0.12) < 1e-6


def test_render_markdown_runs() -> None:
    summary = _mock_summary()
    summary["classification"] = runner.classify_gen2(summary)
    md = runner.render_markdown(summary)
    assert "# AlphaZero-Lite Generation-2 Self-Play Anchor Results" in md
    assert "cumulative_safe_gain" in md
    assert "P1 vs P0 Reproduction" in md
    assert "Generation-2 Candidates vs P1" in md
    assert "Generation-2 Candidates vs P0" in md
