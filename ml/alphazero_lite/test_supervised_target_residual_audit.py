from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    HIDDEN_SIZES,
    MODEL_TYPE,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    input_size_for_encoding,
)
from ml.alphazero_lite import run_supervised_target_residual_audit as audit


def _synthetic_rows(count: int, *, seed: int = 0) -> list[dict]:
    """Generate valid, distinct, round-trippable encoded training rows."""
    rng = np.random.default_rng(seed)
    start = {
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    frontier = [KalahGame.from_state(start)]
    seen: set[tuple] = set()
    rows: list[dict] = []
    while frontier and len(rows) < count:
        game = frontier.pop(0)
        for move in game.possible_moves():
            if len(rows) >= count:
                break
            clone = game.clone()
            if not clone.move(clone.pit_index(move)):
                continue
            state = clone.to_state()
            key = (
                tuple(state["player_pits"]),
                tuple(state["opponent_pits"]),
                state["player_store"],
                state["opponent_store"],
                state["current_player"],
            )
            if key in seen:
                continue
            seen.add(key)
            legal = game.possible_moves()
            policy = np.zeros(6, dtype=np.float64)
            policy[legal] = rng.dirichlet(np.ones(len(legal)))
            rows.append(
                {
                    "state": encode_state(state, input_encoding="kalah_v3"),
                    "policy": policy.tolist(),
                    "value": float(rng.choice([-1.0, 1.0])),
                    "player": int(state["current_player"]),
                    "move_index": int(len(rows)),
                    "action_sampling_noise_enabled": bool(len(rows) % 2 == 0),
                }
            )
            frontier.append(clone)
    return rows


def test_bootstrap_median_ci_reports_median_estimator_ci() -> None:
    values = [float(index) for index in range(1, 101)]
    result = audit.bootstrap_median_ci(values, seed=7)
    assert result["estimator"] == "median"
    assert result["median"] == np.median(values)
    assert result["lower_95"] <= result["median"] <= result["upper_95"]
    repeated = audit.bootstrap_median_ci(values, seed=7)
    assert repeated == result


def test_bootstrap_mean_ci_labels_mean_and_contains_point() -> None:
    values = [float(index) for index in range(1, 101)]
    result = audit.bootstrap_mean_ci(values, seed=8)
    assert result["estimator"] == "mean"
    assert result["mean"] == np.mean(values)
    assert result["lower_95"] <= result["mean"] <= result["upper_95"]


def test_bootstrap_difference_ci_is_direct_and_negative_when_left_below_right() -> None:
    left = [0.1, 0.2, 0.3, 0.4]
    right = [0.6, 0.7, 0.8, 0.9]
    result = audit.bootstrap_difference_ci(left, right, seed=9)
    assert result["statistic"] == "mean"
    assert result["point"] == pytest.approx(np.mean(left) - np.mean(right))
    assert result["upper_95"] < 0.0
    median_result = audit.bootstrap_difference_ci(
        left, right, seed=9, statistic="median"
    )
    assert median_result["point"] == pytest.approx(np.median(left) - np.median(right))


def test_kl_divergence_and_entropy_and_top_move() -> None:
    target = np.asarray([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
    current = np.asarray([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    expected = 0.8 * np.log(0.8 / 0.5) + 0.2 * np.log(0.2 / 0.5)
    assert audit.kl_divergence(target, current) == pytest.approx(expected)
    assert audit.policy_entropy(target) == pytest.approx(
        -(0.8 * np.log(0.8) + 0.2 * np.log(0.2))
    )
    assert audit.top_move(target, [0, 1, 2, 3, 4, 5]) == 0


def test_classify_policy_misaligned_branch() -> None:
    summary = {
        "phase_c_policy_top_move": {
            "forced_continuation_primary": {
                "disagreement_states": 200,
                "by_budget": {
                    "768": {"margin_delta": {"mean": -0.05, "upper_95": -0.01}},
                    "1200": {"margin_delta": {"mean": -0.06, "upper_95": -0.02}},
                },
                "margin_delta_by_policy_residual_quartile": {
                    "0": {"mean": 0.0},
                    "1": {"mean": 0.0},
                    "2": {"mean": 0.0},
                    "3": {"mean": -0.10},
                },
            }
        },
        "phase_e_value_confirmation": {
            "sign_agreement_d1200": {"high_residual_agreement": 0.90}
        },
        "phase_f_off_policy_diagnostic": {
            "split": {
                "replay_matches": {"sign_agreement_d1200": 0.90},
                "replay_differs": {"sign_agreement_d1200": 0.85},
            }
        },
        "phase_g_gradient_attribution": {
            "policy_misaligned_aggregate": {"cosine_with_harmful": 0.4},
            "value_disagrees_aggregate": {"cosine_with_harmful": -0.2},
            "policy_aligned_aggregate": {"cosine_with_harmful": 0.5},
            "value_confirmed_aggregate": {"cosine_with_harmful": 0.5},
        },
    }
    result = audit.classify(summary)
    assert result["label"] == "policy_targets_causally_misaligned"
    assert "policy" in result["next_action"]


def test_classify_value_problematic_branch() -> None:
    summary = {
        "phase_c_policy_top_move": {
            "forced_continuation_primary": {
                "disagreement_states": 10,
                "by_budget": {
                    "768": {"margin_delta": {"mean": 0.0, "upper_95": 0.1}},
                    "1200": {"margin_delta": {"mean": 0.0, "upper_95": 0.1}},
                },
                "margin_delta_by_policy_residual_quartile": {
                    "0": {"mean": 0.0},
                    "1": {"mean": 0.0},
                    "2": {"mean": 0.0},
                    "3": {"mean": 0.0},
                },
            }
        },
        "phase_e_value_confirmation": {
            "sign_agreement_d1200": {"high_residual_agreement": 0.50}
        },
        "phase_f_off_policy_diagnostic": {
            "split": {
                "replay_matches": {"sign_agreement_d1200": 0.90},
                "replay_differs": {"sign_agreement_d1200": 0.60},
            }
        },
        "phase_g_gradient_attribution": {
            "policy_misaligned_aggregate": {"cosine_with_harmful": 0.0},
            "value_disagrees_aggregate": {"cosine_with_harmful": 0.4},
            "policy_aligned_aggregate": {"cosine_with_harmful": -0.1},
            "value_confirmed_aggregate": {"cosine_with_harmful": -0.1},
        },
    }
    result = audit.classify(summary)
    assert result["label"] == "value_targets_high_variance_or_off_policy"


def test_classify_both_problematic_branch() -> None:
    summary = {
        "phase_c_policy_top_move": {
            "forced_continuation_primary": {
                "disagreement_states": 200,
                "by_budget": {
                    "768": {"margin_delta": {"mean": -0.05, "upper_95": -0.01}},
                    "1200": {"margin_delta": {"mean": -0.06, "upper_95": -0.02}},
                },
                "margin_delta_by_policy_residual_quartile": {
                    "0": {"mean": 0.0},
                    "1": {"mean": 0.0},
                    "2": {"mean": 0.0},
                    "3": {"mean": -0.10},
                },
            }
        },
        "phase_e_value_confirmation": {
            "sign_agreement_d1200": {"high_residual_agreement": 0.50}
        },
        "phase_f_off_policy_diagnostic": {
            "split": {
                "replay_matches": {"sign_agreement_d1200": 0.90},
                "replay_differs": {"sign_agreement_d1200": 0.60},
            }
        },
        "phase_g_gradient_attribution": {
            "policy_misaligned_aggregate": {"cosine_with_harmful": 0.4},
            "value_disagrees_aggregate": {"cosine_with_harmful": 0.4},
            "policy_aligned_aggregate": {"cosine_with_harmful": 0.0},
            "value_confirmed_aggregate": {"cosine_with_harmful": 0.0},
        },
    }
    result = audit.classify(summary)
    assert result["label"] == "both_supervised_targets_problematic"
    assert "do not change both" in result["next_action"]


def test_classify_distillation_failure_and_inconclusive() -> None:
    base = {
        "phase_c_policy_top_move": {
            "forced_continuation_primary": {
                "disagreement_states": 10,
                "by_budget": {
                    "768": {"margin_delta": {"mean": 0.01, "upper_95": 0.1}},
                    "1200": {"margin_delta": {"mean": 0.01, "upper_95": 0.1}},
                },
                "margin_delta_by_policy_residual_quartile": {
                    "0": {"mean": 0.0},
                    "1": {"mean": 0.0},
                    "2": {"mean": 0.0},
                    "3": {"mean": 0.0},
                },
            }
        },
        "phase_e_value_confirmation": {
            "sign_agreement_d1200": {"high_residual_agreement": 0.90}
        },
        "phase_f_off_policy_diagnostic": {
            "split": {
                "replay_matches": {"sign_agreement_d1200": 0.90},
                "replay_differs": {"sign_agreement_d1200": 0.85},
            }
        },
    }
    sound = {
        **base,
        "phase_g_gradient_attribution": {
            "policy_misaligned_aggregate": {"cosine_with_harmful": -0.1},
            "value_disagrees_aggregate": {"cosine_with_harmful": -0.1},
            "policy_aligned_aggregate": {"cosine_with_harmful": 0.5},
            "value_confirmed_aggregate": {"cosine_with_harmful": 0.5},
        },
    }
    assert (
        audit.classify(sound)["label"]
        == "targets_individually_sound_objective_distillation_failure"
    )
    inconclusive = {
        **base,
        "phase_g_gradient_attribution": {
            "policy_misaligned_aggregate": {"cosine_with_harmful": -0.1},
            "value_disagrees_aggregate": {"cosine_with_harmful": -0.1},
            "policy_aligned_aggregate": {"cosine_with_harmful": -0.1},
            "value_confirmed_aggregate": {"cosine_with_harmful": -0.1},
        },
    }
    assert (
        audit.classify(inconclusive)["label"] == "target_residual_quality_inconclusive"
    )


def test_build_probe_selects_unique_stratified_states() -> None:
    rows = _synthetic_rows(96)
    source = np.arange(len(rows), dtype=np.int64)
    policies = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    values = np.asarray([row["value"] for row in rows], dtype=np.float64)
    probe, manifest = audit.build_probe(rows, source, policies, values, size=32)
    assert len(probe) == 32
    assert len({row["state_hash"] for row in probe}) == 32
    assert set(manifest["stratum_counts"]) == {
        f"p{qp}_v{qv}" for qp in range(4) for qv in range(4)
    }
    assert manifest["state_count"] == 32
    for row in probe:
        assert set(row["legal_moves"]).issubset({0, 1, 2, 3, 4, 5})
        assert row["policy_residual_quartile"] in range(4)
        assert row["value_residual_quartile"] in range(4)


def test_search_seed_is_treatment_invariant() -> None:
    record = {
        "manifest_index": 3,
        "state_hash": "abc123",
        "player": 1,
    }
    left = audit.search_seed_for_state(record, "manifest-hash")
    right = audit.search_seed_for_state(record, "manifest-hash")
    assert left == right
    other = audit.search_seed_for_state(
        {**record, "manifest_index": 4}, "manifest-hash"
    )
    assert left != other


def test_continuation_experiment_seed_depends_only_on_state_and_budget() -> None:
    a = audit.continuation_experiment_seed("state", 1200)
    b = audit.continuation_experiment_seed("state", 1200)
    assert a == b
    assert a != audit.continuation_experiment_seed("state", 768)
    assert a != audit.continuation_experiment_seed("other", 1200)


def test_per_example_gradients_separates_channels_and_stays_finite() -> None:
    device = torch.device("cpu")
    model = PolicyValueNet(
        HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding("kalah_v3")
    ).to(device)
    rows = _synthetic_rows(8)
    probe = [
        {
            "training_position": index,
        }
        for index in range(4)
    ]
    gradients = audit.per_example_gradients(model, rows, probe, device)
    for signal in ("policy", "value", "joint"):
        tensor = gradients[signal]["shared_trunk"]
        assert tensor.shape[0] == 4
        assert bool(torch.isfinite(tensor).all())
    assert not torch.equal(
        gradients["policy"]["shared_trunk"], gradients["value"]["shared_trunk"]
    )


def test_current_policy_value_masks_illegal_moves() -> None:
    device = torch.device("cpu")
    model = PolicyValueNet(
        HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding("kalah_v3")
    ).to(device)
    state = KalahGame.from_state(
        {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    encoded = encode_state(state.to_state(), input_encoding="kalah_v3")
    x = np.asarray([encoded], dtype=np.float32)
    from ml.alphazero_lite.train import legal_mask_matrix_for_encoded_states

    mask = legal_mask_matrix_for_encoded_states(x)
    policy, value = audit.current_policy_value(model, x, mask, device)
    assert policy.shape == (1, 6)
    assert np.all(policy[0, ~mask[0].astype(bool)] == 0.0)
    assert np.isclose(policy[0].sum(), 1.0)
    assert value.shape == (1,)


def test_markdown_contains_required_sections(tmp_path: Path) -> None:
    summary = {
        "phase_i_classification": {
            "label": "targets_individually_sound_objective_distillation_failure",
            "next_action": "investigate function-space/search-aware constraints",
            "evidence": {
                "policy_disagreement_states": 158,
                "value_high_residual_agreement_d1200": 0.9107,
            },
        },
        "phase_a_pr197_statistical_repair": {
            "phase_f_bootstrap_corrected": {
                "median_estimator": {
                    "median": 0.62,
                    "lower_95": 0.61,
                    "upper_95": 0.63,
                },
                "mean_estimator": {"mean": 0.616, "lower_95": 0.614, "upper_95": 0.618},
            },
            "phase_d_direct_bootstrap": {
                str(k): {
                    "between_minus_within_mean": {
                        "point": -0.1,
                        "lower_95": -0.2,
                        "upper_95": -0.05,
                    },
                }
                for k in audit.EFFECTIVE_BATCH_SIZES
            },
        },
        "phase_b_frozen_probe": {
            "probe_manifest": {
                "state_count": 1024,
                "policy_residual_quartile_boundaries": [0.1, 0.4, 1.0],
                "value_residual_quartile_boundaries": [0.4, 0.7, 1.0],
                "stratum_counts": {"p0_v0": 64},
            },
            "residual_distribution": {
                "policy_residual_kl": {
                    "mean": 0.1,
                    "median": 0.1,
                    "p90": 0.1,
                    "max": 1.0,
                },
                "value_residual_abs": {
                    "mean": 0.1,
                    "median": 0.1,
                    "p90": 0.1,
                    "max": 1.0,
                },
            },
        },
        "phase_c_policy_top_move": {
            "top_move_agreement": {
                "raw_vs_replay": 0.1,
                "raw_vs_d384": 0.1,
                "raw_vs_d1200": 0.1,
                "replay_vs_d384": 0.1,
                "replay_vs_d1200": 0.1,
            },
            "forced_continuation_primary": {
                "disagreement_states": 158,
                "by_budget": {
                    str(b): {
                        "margin_delta": {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0},
                        "outcome_delta": {"mean": 0.0},
                    }
                    for b in audit.PHASE_C_CONTINUATION_BUDGETS
                },
                "margin_delta_by_policy_residual_quartile": {
                    q: {"mean": 0.0} for q in "0123"
                },
            },
        },
        "phase_d_distribution_alignment": {
            "subset_states": 256,
            "alignment": {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0},
            "pairwise_concordance": {"mean": 0.5, "lower_95": 0.5, "upper_95": 0.5},
            "expected_causal_quality_change": {
                "mean": 0.0,
                "lower_95": 0.0,
                "upper_95": 0.0,
            },
        },
        "phase_e_value_confirmation": {
            "sign_agreement_d1200": {"overall": 0.9, "high_residual_agreement": 0.9},
            "sign_agreement_d768": {"overall": 0.9},
            "squared_error_change": {"mean": 0.0},
        },
        "phase_f_off_policy_diagnostic": {
            "match_rate": 0.8,
            "split": {
                "replay_matches": {"sign_agreement_d1200": 0.9, "informative_n": 1},
                "replay_differs": {"sign_agreement_d1200": 0.7, "informative_n": 1},
            },
            "temperature_split": {
                "early_high_temp": {"sign_agreement_d1200": 0.6, "informative_n": 1},
                "later_low_temp": {"sign_agreement_d1200": 0.9, "informative_n": 1},
            },
        },
        "phase_g_gradient_attribution": {
            "policy_aligned_aggregate": {"n": 1, "cosine_with_harmful": 0.5},
            "policy_misaligned_aggregate": {"n": 1, "cosine_with_harmful": -0.1},
            "value_confirmed_aggregate": {"n": 1, "cosine_with_harmful": 0.1},
            "value_disagrees_aggregate": {"n": 1, "cosine_with_harmful": -0.1},
        },
        "phase_h_counterfactual": {
            "G_all": {"cosine_with_harmful": 0.5, "raw_norm": 1.0, "adam_cosine": 0.4},
            "G_policy_aligned_only": {
                "cosine_with_harmful": 0.5,
                "raw_norm": 1.0,
                "adam_cosine": 0.4,
            },
            "G_value_confirmed_only": {
                "cosine_with_harmful": 0.5,
                "raw_norm": 1.0,
                "adam_cosine": 0.4,
            },
            "G_both_filtered": {
                "cosine_with_harmful": 0.5,
                "raw_norm": 1.0,
                "adam_cosine": 0.4,
            },
        },
    }
    report = audit.markdown(summary)
    for section in (
        "PR197 statistical repair",
        "Frozen probe provenance",
        "Policy residual distribution",
        "Policy top-move causal quality",
        "Distribution-level gradient/quality alignment",
        "Value residual confirmation",
        "Exploratory-vs-deterministic value-target analysis",
        "Harmful-gradient attribution",
        "Exact classification evidence",
        "One next action",
    ):
        assert section in report
